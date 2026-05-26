# CLI Tool Mode

Default Webwright runs produce a one-shot `final_script.py` that solves the task for the literal values the user provided. **CLI tool mode** instead produces a **reusable, parameterized CLI tool**: the same script can be re-run later with different argument values to perform the same kind of task.

This mode is adapted from `webwright/config/crafted_cli.yaml`'s "Final-Script Shape (CLI Tool, MANDATORY)" contract from the upstream Microsoft Webwright repo. The OpenAI-backed `self_reflection` gate is replaced by the agent's own self-verification against `plan.md`.

## When to use

Trigger CLI tool mode when:

- the user says "make it reusable", "parameterize", "turn this into a CLI", "I want to call this again with different X", or similar; or
- the spike is meant to become a JurisNet adapter regression harness that another script will import and call with different fallo ids, jurisdictions, or date ranges.

Otherwise, stay in default one-shot mode.

## `plan.md` — add a `# Parameters` section

Before writing the script, identify every requirement the user could plausibly vary and list them in `plan.md` **in addition to** the usual `# Critical Points` checklist:

```markdown
# Task
<verbatim task description>

# Parameters
| name    | type | source phrase from task | default     | allowed / format        |
|---------|------|-------------------------|-------------|-------------------------|
| <arg_a> | str  | "..."                   | "<value>"   | <format / allowed set>  |
| <arg_b> | int  | "..."                   | <value>     | <range or units>        |
| <arg_c> | str  | "..."                   | "<value>"   | <format>                |

# Critical Points
- [ ] CP1: ...
- [ ] CP2: ...
```

Rules:

- Every entry in `# Parameters` must (a) become a function argument and (b) become an `argparse --flag` with the listed default.
- Items that are truly fixed for the site (start URL, site name, selector strategy, proxy env var name) are NOT parameters — keep them hard-coded.
- Defaults reproduce the original task exactly. Running `python final_script.py` with no arguments must reproduce the task.
- Critical Points are still required; they are the verification contract.

## `final_script.py` — required shape

1. **One reusable function** named after the task domain. Examples:
   - `def search_<domain>(arg_a, arg_b, ...): ...`
   - `def lookup_<entity>(query, filters): ...`
   - `def fetch_fallo(fallo_id: str, jurisdiction: str): ...`

2. **Google-style docstring** with summary, full `Args:` block, and `Returns:`. Each `Args:` entry documents:
   - the argument name and type,
   - what it represents in the task domain,
   - accepted format / units / allowed values,
   - the default (mirroring the `# Parameters` table).

   ```python
   def search_<domain>(arg_a: str, arg_b: int, arg_c: str) -> dict:
       """<One-line summary of what this tool does on the target site>.

       Args:
           arg_a: <what it represents>; <format / allowed values>.
               Default: "<value>".
           arg_b: <what it represents>; <range / units>.
               Default: <value>.
           arg_c: <what it represents>; <format>.
               Default: "<value>".

       Returns:
           dict with keys ``<key1>`` (<type>), ``<key2>`` (<type>), ...
       """
   ```

3. **`argparse` CLI** under `if __name__ == "__main__":`. Every function argument has a matching `--<arg>` flag with `type=`, `help=` (copied from the docstring), and `default=` equal to the concrete task value:

   ```python
   if __name__ == "__main__":
       import argparse
       parser = argparse.ArgumentParser(
           description=search_<domain>.__doc__.splitlines()[0])
       parser.add_argument("--arg-a", dest="arg_a", type=str,
                           default="<value>",
                           help="<copied from docstring>")
       parser.add_argument("--arg-b", dest="arg_b", type=int,
                           default=<value>,
                           help="<copied from docstring>")
       parser.add_argument("--arg-c", dest="arg_c", type=str,
                           default="<value>",
                           help="<copied from docstring>")
       args = parser.parse_args()
       result = asyncio.run(_run(**vars(args)))
       print(result)
   ```

4. **Side-effect-free at import time.** No browser launch, no network call, no file write at module top-level. The reusable function must be importable from another Python process without triggering a run.

5. **Action-log parameter echo.** The first line written to `final_script_log.txt` after reset MUST be a `step 0 params: ...` line listing every resolved argument as `name=value` pairs, e.g.:

   ```
   step 0 params: arg_a=<value> arg_b=<value> arg_c=<value>
   ```

   so the resolved inputs are visible in any verification pass.

6. Same instrumentation as default mode: viewport 1280×1800, headless local Firefox by default, no `full_page=True`, screenshots saved as `final_runs/run_<id>_<tag>/screenshots/final_execution_<step>_<action>.png`, final datum appended to `final_script_log.txt`, `results.json` written at the end.

## Verification (replaces `self_reflection`)

In addition to the default self-verification (every CP in `plan.md` ticked with cited screenshot/log evidence), CLI mode requires:

1. **Reproduce the task with no arguments.** Inside a fresh `final_runs/run_<id>_<tag>/`:

   ```bash
   cd final_runs/run_<id>_<tag> && python final_script.py
   ```

   The run must succeed end-to-end and produce the expected screenshots and `step 0 params: ...` log line.

2. **Import-safety smoke test.** From any other directory:

   ```bash
   python -c "import importlib.util, pathlib; \
     spec = importlib.util.spec_from_file_location('fs', 'final_runs/run_<id>_<tag>/final_script.py'); \
     m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
     print([n for n in dir(m) if not n.startswith('_')])"
   ```

   This must complete instantly with no browser launch and print the reusable function's name.

3. **Optional second run with a different argument value.** Demonstrates parameterization actually works. Run inside `final_runs/run_<id+1>_alt/` (or save its log/screenshot folder there). Skip only if the alternate value would clearly fail (e.g. an unsupported value on the target site).

4. **Print `--help`.** End by showing the user:

   ```bash
   python final_runs/run_<id>_<tag>/final_script.py --help
   ```

## Completion gate (CLI mode)

Set the task complete only when **all** are true:

1. `plan.md` contains both `# Parameters` (with name, type, source phrase, default, allowed/format) and `# Critical Points` checklists.
2. `final_script.py` defines exactly one reusable function with a Google-style `Args:` docstring covering every parameter.
3. Every `# Parameters` entry maps 1-to-1 to a function argument **and** an argparse `--flag` whose default equals the concrete task value.
4. The script is import-safe (smoke test passes — no browser launch, no I/O on import).
5. `python final_script.py` (no args) inside `final_runs/run_<id>_<tag>/` reproduced the task; all CPs verified against saved screenshots and the action log.
6. `step 0 params: ...` line is present in `final_script_log.txt`.
7. The user has seen the final datum **and** the `--help` output so they know how to call the tool again with different arguments.

If any of those is false, do not declare done — diagnose, fix the script (preserving the CLI shape), re-run inside the next `run_<id+1>_<tag>/`, and re-verify.
