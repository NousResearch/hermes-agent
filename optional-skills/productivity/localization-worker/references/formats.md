# Verified Format Matrix

| Extensions | Inspection | Extraction | Round trip | Verified guarantees |
|---|---:|---:|---:|---|
| `.txt` | yes | physical line units | yes | UTF-8 reopen, BOM, LF/CRLF, final newline, exact validated text |
| `.md` | yes | physical line units | constrained | same text-framing guarantees as `.txt`; Markdown AST semantics are not parsed |
| all other extensions | explicit unsupported | no | no | `UNSUPPORTED_FORMAT` |

Support is declared only where native-plugin round-trip tests exist. The plugin never silently flattens an unsupported structure.

## Current limits

- UTF-8 only. Invalid byte sequences fail closed.
- Newlines must be uniformly LF or uniformly CRLF. Bare-CR, mixed styles, U+0085, U+2028, U+2029, vertical tab, and form feed fail with `UNSUPPORTED_NEWLINE_STYLE`.
- Markdown is line-preserving text. Front matter, fenced code, links, inline HTML, and other syntax are not independently parsed or protected.
- Empty files fail with `EMPTY_DOCUMENT`.
- A single line above the 2,048 estimated-token source budget fails with `OVERSIZED_SEGMENT`.
- Multiple lines are grouped into deterministic chunks below that budget.
- Output paths are plugin-owned and profile-scoped.
- Input and output artifacts are opened component by component with POSIX directory fds plus no-follow and nonblocking final-open semantics; traversal, ancestor replacement, symlink escapes, FIFOs, sockets, devices, and other non-regular files are rejected without blocking.
- Extraction returns bounded metadata. Source text is returned only for segments in a valid claimed chunk lease.
- Linux and macOS are supported. The profile home and SQLite plugin-data directory are an administrator-owned trust boundary.

## Planned adapters

CSV, TSV, JSON, JSONL, YAML, properties, subtitles, Office, PDF, CAT, XML, HTML, and software-resource formats require dedicated inspect/extract/assemble/reparse/validate adapters and representative fidelity tests before they can move out of `UNSUPPORTED_FORMAT`.
