**When to load:** the user wants EGM-style contract READMEs (interface, invariants, side effects) created or corrected across an existing directory tree.

# EGM Peanuts and Hay Bootstrap (the bootstrap workflow)

## Overview

This workflow automates the creation, correction, formatting, and correctness
verification of Elephant-and-Goldfish Model (EGM) compliant `README.md` files
across a directory tree. It establishes clear architectural contracts
(interfaces, invariants, and side effects) for both human developers and AI
assistants.

*   **Peanuts**: Leaf-level Component Package READMEs containing specific file
    contracts.
*   **Hay**: Parent-level Repository Blueprints that roll up child interface
    contracts into high-altitude indexes.

## Delegation

A Goldfish runs in a fresh context with no conversation history. In Hermes,
spawn each Goldfish with `delegate_task`. Each reviewer persona lives in
`references/reviewers/<file>.md` of the `elephant-goldfish` skill; its persona
id (for example `egm-zoom-out`) is recorded in the comment on the file's first
line. Load the persona text with
`skill_view(name='elephant-goldfish', file_path='references/reviewers/<file>.md')`
and paste it verbatim into the task `context`, together with ONLY: the absolute
path of the document under review, the line range, the referenced source paths
(`GoldfishContext`), the applicable document standard text, the `Mode`, and the
required return format. Never put conversation history, your own reasoning, or
the user's earlier discussion into a Goldfish `context`. Add the line
"Do not modify any files; report findings only." Hermes does not enforce
read-only reviewers; that line is the only guard.

Run several reviewers in parallel as ONE `delegate_task` call with a `tasks`
array (one entry per reviewer). Hermes delegation is flat by default (children
cannot spawn their own children unless `delegation.max_spawn_depth` is 2 or
more), so use direct dispatch: spawn the reviewers yourself, then run the
synthesizer as a separate `delegate_task` fed only the reviewer outputs and
the document path. See the hub `SKILL.md`, section "Spawning goldfish in Hermes".
If `delegate_task` is unavailable, run the scout steps inline.

In this workflow the Goldfish are the sub-orchestrator workers and the
`goldfish-explorer` scouts, not reviewers. A worker can spawn its own scouts
only when `delegation.max_spawn_depth` is 2 or more; under the flat default,
each worker runs its scout steps inline.

## Inputs

*   **`path`**: The target root directory path to bootstrap (Required).
*   **`is_worker`**: Internal parameter. `true` if executing as a worker
    Sub-Orchestrator (Optional, default: `false`).
*   **`sub_tree_root`**: Internal parameter. The root path of the assigned
    subtree partition (Optional).
*   **`target_filename`**: Target contract filename to generate or update
    (Optional, default: `"README.md"`). Supports alternate blueprint filenames
    such as `"EGM.md"` when pre-existing human-maintained READMEs must not be
    modified.

## Execution Constraints (CRITICAL)

1.  **No Code Generation**: You are strictly limited to documentation. Do not
    write or refactor any functional source code.
2.  **VCS Isolation**: Do not perform any version control commits (`git commit`,
    `git add`). All filesystem changes must be written locally using standard
    file-writing tools. Tracking
    and submitting is deferred to the human user.
3.  **Temporary File Isolation**: Parallel workers sharing the workspace must
    perform all status tracking, check-pointing, and log writing in independent,
    worker-specific directories outside the project repository workspace.
4.  **Tool Usage (CRITICAL)**: Use `terminal`, `read_file`, `search_files`, `write_file`, and `patch` for directory discovery and contract/README
    generation. Adhere to clean Markdown formatting directly during generation
    (standard headers, tables, and fenced code blocks). If a project markdown
    formatter (e.g. `npx prettier --write`) is explicitly configured in the
    repository, it may optionally be invoked.
5.  **Subagent Delegation**: Spawn sub-orchestrator workers in parallel.
6.  **Mandatory Safety Gate (CRITICAL)**: If you discover more than 15
    subdirectories, you MUST immediately request user approval (using the `clarify` tool, or plain chat) before proceeding with full-tree execution
    or unbounded subagent spawning.

---

## Workflow

### Step 0: Ecosystem & Verification Discovery

1.  Inspect the repository root for package/build manifests:
    *   Node.js / TypeScript: `package.json` -> `npm test` (or `pnpm test` / `yarn test`)
    *   Rust: `Cargo.toml` -> `cargo test`
    *   Go: `go.mod` -> `go test ./...`
    *   Python: `pyproject.toml` / `pytest.ini` / `setup.py` -> `pytest`
    *   C/C++ / Make: `Makefile` -> `make test`
    *   Bazel: `BUILD` / `BUILD.bazel` -> `bazel test //...`
    *   None found: `<test-command>`
2.  Record as `VerificationCommand` to populate generated README contracts.

---

### Step 1: Startup Branching Control

1.  Check the `is_worker` input parameter.
2.  If `is_worker` is `true`, jump directly to **Step 4: Sub-Orchestrator Worker
    Flow** to process the assigned `sub_tree_root`.
3.  If `is_worker` is `false` or unset, proceed as the Root Orchestrator to
    **Step 2: Incremental Scoping & Safety Gate**.

---

### Step 2: Incremental Scoping & Safety Gate (Root Orchestrator)

1.  **Incremental BFS Discovery**: Map the target `path` tree level-by-level
    (BFS) using a directory queue, listing one directory level at a time. (CRITICAL: never run a recursive search such as `find` or `ls -R`.)

    *   Preserve intermediate folder nodes.
    *   Ignore symbolic links to prevent infinite recursion.
    *   Ignore hidden/system directories (`.git`, `.hermes`, `node_modules`,
        `venv`, `dist`, `build`, `third_party`).
    *   Read `.gitignore` and `.egmignore` (if present) and ignore matching glob patterns.
    *   Preserve package and build manifests (`package.json`, `Cargo.toml`,
        `go.mod`, `pyproject.toml`, `Makefile`, `BUILD`) in the discovery map
        to extract verification targets later.

2.  **Safety Gate & Unbounded Spawning Approval (CRITICAL)**: During the BFS
    loop, if the total running count of discovered directories exceeds **15**,
    immediately pause discovery and request user approval (using the `clarify` tool, or plain chat) to warn the user and request approval
    for full-tree execution:

    *   `question`: "The target directory tree contains more than 15
        subdirectories. This is a large execution scope. Do you want to proceed
        with full-tree bootstrapping?"
    *   `options`: `["Yes, proceed with full tree bootstrapping",
        "No, abort execution"]`
    *   (CRITICAL: After requesting approval, you MUST wait for the user's
        response before continuing or spawning any workers).
    *   If the user selects abort, terminate execution immediately.

3.  **Bottom-up Post-Pruning Pass**: Once mapping is complete (and approved),
    walk the mapped tree bottom-up (leaves to root). Filter out non-code and
    test-only directories:

    *   *Source Code Files*: Files matching recognized source code, schema, and
        contract extensions:
        - Systems & Compiled: `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hh`, `.hpp`,
          `.rs`, `.go`, `.zig`, `.lean`
        - Managed & Enterprise: `.java`, `.kt`, `.scala`, `.cs`, `.fs`
        - Scripting & Dynamic: `.py`, `.rb`, `.php`, `.lua`, `.sh`, `.bash`
        - Web & Mobile: `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs`, `.swift`,
          `.m`, `.dart`
        - Data & Schemas / IDLs: `.sql`, `.proto`, `.graphql`, `.gql`,
          `.thrift`, `.prisma`
        - Infrastructure / Declarative Code: `.tf`, `.hcl`, `.cue`
        (Excludes build manifests, lockfiles, test fixtures, documentation, and
        generic data dumps like `.csv`, `.tsv`, `.parquet`).
    *   *Test Directory Exclusion*: Exclude directories containing exclusively
        test code (e.g. named `tests`, `testing`, `testdata`, or containing only
        files matching `*_test.*`, `*Test.*`, `test_*.*`, or `*.spec.*`).
    *   *Non-Code Filtering*: Exclude any directory that contains no source code
        files AND contains no subdirectories with code. Retain empty
        intermediate folders only if they have child directories that contain
        code.

4.  Calculate the total count of pruned directories and maximum depth. Proceed
    to **Step 3: Execution Strategy Selection**.

---

### Step 3: Execution Strategy Selection (Root Orchestrator)

1.  **Direct Execution Strategy (Directory Count <= 5)**:

    *   If the total count of pruned directories is 5 or fewer, execute the bottom-up
        documentation directly in the current thread.
    *   Jump to **Step 4: Sub-Orchestrator Worker Flow** (treating the Root
        Orchestrator as the worker for the entire `path`), and then proceed to
        **Step 5: Root Compilation & Termination**.

2.  **Unbounded Parallel Subagent Swarm Strategy (Directory Count > 5)**:

    *   (CRITICAL: If Directory Count > 15, you MUST verify that you have
        received explicit user approval before proceeding. If not yet approved,
        request approval immediately and wait for the user's response before
        spawning workers).
    *   Partition the discovery map into independent subtree branches using the
        **Adaptive Depth Partitioning Algorithm**:
        1.  Set `depth = 1`.
        2.  Extract directory nodes from the discovery map located at the
            current `depth`.
        3.  If the count of nodes at `depth` is between 3 and 5: select these
            nodes as subtree partition roots.
        4.  If the count of nodes at `depth` < 3: increment `depth` (up to a max
            of 3) and repeat step 2.
        5.  If the count of nodes at `depth` > 5: group the largest branches (by
            total nested folder count) into 3-4 separate partitions, combine the
            remaining small branches into a single fallback partition subtree,
            and select these as partition roots.
        6.  Fallback: If `depth` reaches 3 and count is still < 3, select the
            immediate children of the root (or fall back to sequential
            execution).
    *   **Spawn Unbounded Sub-Orchestrators**: For each partition subtree root,
        spawn a parallel worker with `delegate_task` (one `tasks` entry per partition, all in ONE call; pass the contents of `templates/worker-return.schema.json` as the task's `output_schema`):
        *   Subagent: `egm-bootstrap-orchestrator` (`references/reviewers/bootstrap_orchestrator.md`) (paste its text into the worker `context`)
        *   `Prompt`: "Load `skill_view(name='elephant-goldfish', file_path='references/workflows/bootstrap.md')` and run the bootstrap workflow with path=`[path]`,
            is_worker=true, sub_tree_root=`[partition_root]`,
            target_filename=`[target_filename]`."
    *   **Unbounded Execution**: Spawn all Sub-Orchestrators in parallel without
        artificial concurrency limits (as authorized by the user during the
        safety gate).
    *   Go idle and wait for all worker subagents to return their JSON subtree
        index payloads. Jump to **Step 5: Root Compilation & Termination**.

---

### Step 4: Sub-Orchestrator Worker Flow

1.  **Scout Registration**: Use a read-only `goldfish-explorer` scout. Hermes has no runtime subagent registration: spawn each scout as a `delegate_task` task whose `goal` is `goldfish-explorer scout for <D>` and whose `context` carries the `system_prompt` below verbatim, the files to read, and the line "Do not modify any files and do not spawn subagents." Optionally pass `{"interface": [], "invariants": [], "side_effects": []}` as an `output_schema`. The two `enable_*` flags below are NOT enforceable in Hermes; they are instructions only:

    *   `name`: `goldfish-explorer`
    *   `description`: `Read-only codebase explorer scout.`
    *   `system_prompt`: >- You are a read-only codebase explorer. Your task is
        to analyze the source files in the target directory and return a
        concise, factual summary. Do not write or modify any files. Return a
        JSON object containing the directory's public symbols (interface),
        invariants, and side effects matching schema: {"interface": [],
        "invariants": [], "side_effects": []}. Never infer RPC methods,
        routes, status enums, or invariants from file or directory names
        alone; cite the exact `[file:symbol_or_flag]` for every entry, and
        output `["None"]` if no explicit invariants or side effects exist in
        code.
    *   `enable_write_tools`: `false`
    *   `enable_subagent_tools`: `false`

2.  **Post-Order Traversal**: Walk the assigned `sub_tree_root` (or `path`) in
    post-order (deepest leaves first). For each directory `D`:

    *   **Correction & Modification Policy**: Check if `target_filename` (default:
        `README.md`) already exists in `D`.

        *   If `target_filename` exists, **do not overwrite it completely**. Instead,
            read its existing content.
        *   *Scout Scan Requirement*: Whether `target_filename` exists or not,
            **always scan** the source code using the `goldfish-explorer` scout
            for `D` to compile the true code symbol boundary.
        *   *Legacy Extraction & Correction*: If `target_filename` exists, apply this extraction prompt yourself (no separate model call): > "Analyze the following unstructured
            documentation content. Extract all public symbol names, classes,
            interfaces, invariants, and side effects. Format your response
            strictly as a JSON object matching the following schema: {
            \"interface\": string[], \"invariants\": string[], \"side_effects\":
            string[] }." Merge these extracted symbols with the scout's code
            symbols. Use `patch` (or `write_file`) to correct
            and update the existing contract file to be EGM compliant while preserving
            developer notes and descriptions.

    *   **Scout Spawning & Constraints**: Spawn `goldfish-explorer` for `D`.

        *   *Scout Constraints*: No per-directory or per-file byte cap; every
            non-test file in `D` is read in full. Bound each scout call to
            about **1,500 lines** of source instead: list top-level definitions
            (class/func/def/route/flag lines) with `search_files`, cut the
            directory's files into chunks at those boundaries (never mid-body;
            small files share a chunk), spawn one `goldfish-explorer` per
            chunk in parallel, and merge their JSON. Give each chunk the
            file's imports and top-level definition list so it can judge
            visibility. Check build-manifest deps and `import` / `#include`
            statements before attributing an RPC service, schema, or message
            type to a local file. Read non-test files in this order:
            1.  Contracts, schemas, and build manifests (`.h`, `.hh`, `.hpp`,
                `.proto`, `.graphql`, `.gql`, `.thrift`, `.prisma`, `.sql` DDL,
                interface classes); extract exact DDL table names and exported
                build targets.
            2.  Module entry points (`main.*`, `index.*`, `lib.*`, `mod.*`,
                `__init__.py`, or files matching directory name), then
                production source/query files alphabetically.
            3.  Templates and configs: up to 200 lines each (800 in config-only
                directories); exclude `.css`/`.scss`.
            4.  Test files (`*_test.*`, `*Test.*`, `test_*.*`, `*.spec.*`):
                ignore entirely, including in hybrid directories.
        *   *Child Fault Tolerance*: If the scout fails or times out, log a
            warning, use placeholder values `["(Details pending/documentation
            failed)"]`, and continue execution.

    *   **Parent Context Propagation**: When processing a parent directory `P`,
        retrieve the compiled interface summaries of its immediate child
        subdirectories.

        *   *Propagation Heuristics*:
            1.  If the child was documented during this run, retrieve its public
                interface directly from the in-memory execution cache.
            2.  If the child was corrected/modified (Correction Policy), use the
                merged legacy extraction symbols.
        *   Pass these child symbol blocks as prompt context when invoking `P`'s
            explorer scout.

    *   **Verification Command Target Lookup**: Populate the Verification section
        using the discovered `VerificationCommand`.
        *   If the project uses Bazel/BUILD files: check if a `BUILD` file exists
            in `D` or the nearest ancestor, referencing `bazel test` / `bazel build`
            with valid target patterns (`//[package_path]/...` or
            `//[package_path]:all`, never `//[package_path]...` or
            `//[package_path]:...`). Use `bazel run` for interactive runner
            targets.
        *   If the project uses other manifests (`package.json`, `Cargo.toml`,
            `go.mod`, `pyproject.toml`, `Makefile`), populate with the corresponding
            test runner command.
        *   If no verification command is identified, omit or use a placeholder.

    *   **Directory Classification & Template Generation**: Generate or correct
        `target_filename` using `write_file` or `patch` based on
        `D`'s classification.

        *   **Component Package (Peanut)**: `D` contains direct source code
            files. Use the **Component Package README Template**. (If it also
            has child subdirectories, include relative links to their contracts
            in the Interface section).
        *   **Repository Blueprint (Hay)**: `D` contains no direct source code
            files, but contains subdirectories with code contracts. Use the
            **Intermediate Blueprint README Template**.
        *   **Link Formatting Policy (CRITICAL)**: Package contract files must use a
            hybrid link formatting approach to balance portability and clarity:
            1.  **Relative Links to Existing Files Only**: Use relative paths
                (e.g., `./child/${target_filename}` or `../sibling/${target_filename}`) for child
                contracts and sibling file references within the same codebase
                subsystem. (CRITICAL: Only generate `[Child](./child/${target_filename})`
                links for child directories that exist in the pruned
                `subtree_index` and contain an actual `${target_filename}`. If a child
                directory contains a `SKILL.md` or other non-contract asset, link
                directly to `./child/SKILL.md` or format it as an unlinked
                bullet point. Never output speculative links to non-existent
                contract files).
            2.  **Absolute Links**: Use standard URLs or repository paths
                for referencing files outside the package subsystem (e.g.,
                cross-package dependencies).

    *   **Markdown Formatting**: Immediately after writing or correcting
        `target_filename` in `D`, ensure clean markdown formatting (valid headers,
        tables, and code fences). If a project markdown formatter (e.g. `npx prettier --write`)
        is configured in the repository, you may execute it on `target_filename`.

3.  **Worker Reporting**: Once the assigned subtree is fully traversed, compile
    the `subtree_index` array containing the `path`, `summary`, `interface`,
    `invariants`, and `side_effects` for every processed directory.

4.  If `is_worker` is `true`, read the [Worker Return Interface Schema](../../templates/worker-return.schema.json)
    (`skill_view(name='elephant-goldfish', file_path='templates/worker-return.schema.json')`), then return a matching JSON payload to
    the Root Orchestrator and terminate. If
    `is_worker` is `false` (sequential mode), proceed directly to **Step 5: Root
    Compilation & Termination**.

---

### Step 5: Root Compilation & Termination (Root Orchestrator)

1.  **Receive Worker Payloads**: Collect the JSON `subtree_index` payloads
    returned by all parallel Sub-Orchestrators.

    *   **Root Fault Tolerance**: If a parallel worker fails or times out, log a
        warning, insert a placeholder warning link (`[Subtree Root]
        (Documentation failed / Subagent execution timeout)`) in the root
        Blueprint's subdirectory index, and complete the remaining compilation.
        (Worker failures leaving partial files are handled as an incremental
        progress feature; the Correction Policy allows resuming on subsequent
        runs).

2.  **Intermediate Directory Documentation**: Identify any intermediate
    directories sitting between the root `path` and the partitioned subtree
    roots.

    *   Process these intermediate directories sequentially bottom-up.
    *   Generate Intermediate Blueprint contract files using the compiled summaries
        returned by the workers.
    *   Ensure clean formatting and valid relative links for each Intermediate
        Blueprint contract file.

3.  **Root Directory Documentation**: The target root `path` of the execution is
    *always* classified as the **Repository Root (Blueprint / Hay)**.

    *   Check if `target_filename` already exists in `path`. If present, apply the
        **Correction & Modification Policy** to read and retain the existing
        title, metadata, and architectural overview at the top of the root
        contract file.
    *   Check if `path` itself contains any direct source files. If present,
        spawn a `goldfish-explorer` scout for `path` to extract local symbols
        and invariants, appending them under the optional `## Local Interface &
        Invariants` section.
    *   Compile the final root `target_filename` using the **Repository Root Blueprint
        Template**, rolling up all top-level subdirectory boundaries. Ensure any
        root architecture or sequence diagrams strictly match the synchronous /
        asynchronous boundaries, operation names, and storage backends in the
        child `subtree_index` payloads without inventing unverified nodes.
    *   Write the root `target_filename` using `write_file`.

4.  **Final Correctness Verification (CRITICAL)**: You MUST perform a final
    verification pass across ALL created and modified contract files in the
    `subtree_index` (including all child package contracts like `pkg1/${target_filename}`
    and `pkg2/${target_filename}`) as a correctness check before concluding work. You
    MUST explicitly read every single created or modified contract file
    to verify its content. Do not skip any child contract files. Verify:

    *   **EGM Compliance & Code Grounding**: Ensure `Interface`, `Invariants`,
        and `Side Effects` sections exist, cite real code symbols/flags/tables
        from the scout extraction, and contain zero speculative or unverified
        claims.
    *   **Markdown Syntax**: Ensure clean headers, valid tables, and proper code block fences.
    *   **Valid & Resolvable Links (CRITICAL)**: Ensure all subdirectory and
        sibling links use valid relative paths (`./subdir/${target_filename}` or
        `./subdir/SKILL.md`), and explicitly verify that every referenced target
        file actually exists on the filesystem. Remove or correct any broken
        relative references.
    *   **Valid Targets**: Verify verification targets match project test
        configuration and use valid target syntax (for Bazel, `/...` or
        `:all`, never `:...`).

5.  **Optional Drift Review (CRITICAL)**: After verification, ask the user (using
    the `clarify` tool, or plain chat) to offer running a drift report
    on the generated documentation updates:

    *   `question`: "Do you want to run the EGM Drift Report on the generated
        contract updates to check for consistency with the codebase?"
    *   `options`: `["(Recommended) Yes, run drift report on modified contracts",
        "No, skip drift review"]`
    *   If the user selects "Yes", run the drift-report workflow targeting
        the modified contracts to verify both missing code coverage and
        doc-to-code hallucinations (verifying every documented method, route,
        enum, database table, and invariant against local source files and
        recording discrepancies in the drift report for human triage). (Note:
        While the drift-report workflow is often used for design documents, it can be
        applied to EGM-compliant contracts as a variant).

6.  Terminate execution successfully.

---

## Templates

Before writing any contract file, read the
[README templates](../../templates/bootstrap-readme-templates.md)
(`skill_view(name='elephant-goldfish', file_path='templates/bootstrap-readme-templates.md')`).
It defines the **Component Package (Peanuts)**, **Repository Root (Blueprint /
Hay)**, and **Intermediate Blueprint** README templates referenced above.
