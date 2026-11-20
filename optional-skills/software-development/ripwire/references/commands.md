# ripwire — commands reference (measured)

Numbers below were measured live 2026-09-14 on 4 vCPU, ripwire v0.6.0 (built from d7b6c82).

| Repo | Files | Full ranked map | One query (warm cache) |
|------|-------|-----------------|------------------------|
| private trading repo | 519 | 1.0s | 0.1s |
| private work repo | 905 | 1.4s | ~0.1s |
| ripwire itself | 2033 | ~6s | <1s |

## Output format

Minified XML with leading `<!-- legend -->` comments that define every attribute. Key
attributes on ranked rows: `k=` rank, `cx=`/`ccx=` complexity, `in=` reuse count,
`churn=` recent commits, `amp=` change amplification, `tested=1` when an indexed test
transitively reaches the symbol. Every listing discloses `shown=`/`total=`/`capped=`
when a budget cut it — page with `--offset`.

## Flag pitfalls (hit live)

- `--top-k=0` without a payload verb (`--expand`/`--outline`/`--pack-signatures`) errors.
  Use `--top-k=1` for the smallest map.
- `--top-k` narrows the ranked map only; `--impact`/`--callers`/`--uses` take `--limit=N`.
- `--for` self-limits its bundle via `--pack-top-n`; widen a `--for` answer with `--limit`.
- The legend comments are long. In pipelines, strip them:
  `ripwire . --callers=x 2>/dev/null | grep -v '^<!--'`.

## Verbs by question

| Question | Verb |
|----------|------|
| What does this task touch? | `--for="task"` |
| Who calls this function? | `--callers=SYM` (distinct) / `--uses=SYM` (call sites) |
| What breaks if I change it? | `--impact=SYM` (transitive, tested/untested split) |
| Which tests should run? | `--test-gate f1 f2` / `--exercises=TESTFILE` |
| Show me the body | `--expand=PATH:SYMBOL` |
| Is this symbol dead? | `--safe-delete=SYM` (checks reach) |

## Security scan mode

`ripwire --scan-skills <dir>` scans skill files for injection/exfiltration patterns.
Measured on a 376-file skills tree: 67 findings, ALL false positives from vendored
Cloudflare API docs whose curl examples contain `Bearer ***` placeholders. Use it as a
first-pass triage after syncing third-party skills; treat findings in authored SKILL.md
files as signal, findings inside `references/` vendored docs as noise until triaged.

## Rebuild from source

```bash
git clone --depth 1 https://github.com/redhat-et/ripwire.git /tmp/ripwire
cmake -S /tmp/ripwire -B /tmp/ripwire/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ripwire/build -j
/tmp/ripwire/build/ripwire --version
```

Needs GCC 13+ (C++23). ~4 min on 4 vCPU; the source tree with build artifacts is ~520MB,
so build in /tmp and keep only the binary.
