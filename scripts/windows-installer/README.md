# Windows installer authoring

`scripts/install.ps1` is the generated, standalone distribution artifact. Its
path and one-file format are compatibility contracts: released desktop and
bootstrap binaries download that exact raw GitHub path and cache only that file.
The Python updater also copies it into old bootstrap caches. Keep these consumers
working by editing the sources here and regenerating the artifact:

```console
python scripts/build_windows_installer.py
python scripts/build_windows_installer.py --check
```

The assembler uses only Python's standard library. It emits UTF-8 without a BOM,
with LF line endings, on every host. The existing GUI cache writers retain their
own BOM handling for Windows PowerShell 5.1. Source CRLF is normalized to LF; bare
CR, missing final newlines, and source BOMs are rejected. `--check` compares exact
artifact bytes and does not write files.

## Source graph

`manifest.json` names the entry source and every file in `source/`, in depth-first
inclusion order. Source files are literal PowerShell text. A directive on its own
line inserts another source file's text at build time:

```powershell
# @include repository.ps1
```

Include paths are relative to `source/`, including in nested fragments. No
directory traversal, absolute paths, links, duplicate inclusion, cycles, missing
files, unlisted files, orphaned files, or case-colliding manifest paths are allowed.
The reserved `# @include` prefix must appear only as an assembly directive, never
as literal data in a PowerShell here-string.

Assembly performs no runtime module import, scriptblock wrapping, encoding,
compression, or minification. It preserves the original parameter block,
initialization order, function bodies, script scope, stage protocol, entry-point
dispatch, and invocation path. No authored file is loaded from the network or
dot-sourced by the generated installer.

## Extracting one coherent shard

1. Move a contiguous, complete source region verbatim into a topical file in
   `source/`. Preserve all bytes after LF normalization, including comments and
   blank lines; do not split a function or a PowerShell here-string.
2. Replace that exact region with one include directive. Update `files` to match
   depth-first traversal of the resulting inclusion graph.
3. Lower any affected kill-track ceiling to the residual's current physical line
   count, including its include directives. Remove a kill-track entry once that
   source is at or below 2,000 lines.
4. Regenerate and prove the distribution artifact is byte-identical to the
   preceding extraction's artifact. Run the installer behavior checks in both
   Windows PowerShell 5.1 and PowerShell 7.

One coherent extraction belongs in one PR. Behavior changes belong in later PRs
that edit the appropriate authored shard and regenerate the artifact.

## Source size and the temporary residual

The assembly mechanism initially preserves the existing 5,082-line authored
installer in `source/install.ps1`. It remains explicitly listed in `kill_track`;
introducing a generator alone does not complete its sharding. Its ceiling must
decrease with each extraction, and the kill track must be empty when the source
sharding finishes. Every other authored source has a 2,000-line ceiling.

The distribution artifact is a build output only while exact regeneration passes.
Do not create a blanket PowerShell exemption or use this mechanism to hide an
oversized authored input. CI must run the generation check on source, manifest,
assembler, and artifact changes, and must reject increases to kill-track ceilings.
The assembler validates the current graph. CI also passes `--base-ref` and
`--head-ref` to check every manifest-changing commit in the PR, including a
stack of extractions. The first manifest must preserve its parent installer's
source bytes and exact line ceiling; later ceilings may only decrease or be
removed. Removed allowances cannot be reintroduced. CI checks the real PR head
for this history contract and the test-merge checkout for artifact regeneration.

The initial source is taken from Git blob
`e67300b455a1c7d67bc71a94e9a08b04b67f3425:scripts/install.ps1`:
246,939 bytes, SHA-256
`55b3d76e62abba5cecc16d17a65d799aaa6490e3dc88640fdcae282c4db5fd9f`.
Assembly metadata lives here, so the generated script needs no header changes.

The extraction base includes fangliquanflq's original-generation retry recovery in [PR #103771](https://github.com/NousResearch/hermes-agent/pull/103771), following Axl Ibiza's investigation and reproduction in [issue #103751](https://github.com/NousResearch/hermes-agent/issues/103751). This assembly mechanism preserves that implementation verbatim.

This source decomposition follows Axl Ibiza's graph-gated extraction work tracked in [#79922](https://github.com/NousResearch/hermes-agent/issues/79922) and [#78647](https://github.com/NousResearch/hermes-agent/issues/78647). Individual source moves preserve prior contributor code; they do not transfer its authorship.
