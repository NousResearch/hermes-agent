#!/usr/bin/env bash
# Copy this crate's src/ + Cargo.lock into a hermes-agent crates/tui tree.
# Never overwrites the landing `repository` URL.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${1:-}"
if [[ -z "$dest" ]]; then
  echo "usage: $0 /path/to/hermes-agent/crates/tui" >&2
  exit 2
fi
if [[ ! -f "$dest/Cargo.toml" ]]; then
  echo "sync-landing: $dest is not a crate (missing Cargo.toml)" >&2
  exit 2
fi
if grep -q 'repository = "https://github.com/0xNyk/hermes-tui"' "$dest/Cargo.toml"; then
  echo "sync-landing: refusing to clobber a standalone Cargo.toml at $dest" >&2
  exit 2
fi

repo="$(grep -E '^repository =' "$dest/Cargo.toml" || true)"
rsync -a --delete "$root/src/" "$dest/src/"
cp "$root/Cargo.lock" "$dest/Cargo.lock"

python3 - "$root/Cargo.toml" "$dest/Cargo.toml" "$repo" <<'PY'
import re, sys
src, dest, repo_line = sys.argv[1], sys.argv[2], sys.argv[3]
src_txt = open(src).read()
dst_txt = open(dest).read()
def field(name, text):
    m = re.search(rf'^{name} = .*$', text, re.M)
    return m.group(0) if m else None
for name in ("rust-version",):
    s, d = field(name, src_txt), field(name, dst_txt)
    if s and d:
        dst_txt = dst_txt.replace(d, s, 1)
def block(text, start, end):
    a = text.find(start)
    b = text.find(end, a)
    return text[a:b] if a >= 0 and b >= 0 else None
src_deps = block(src_txt, "[dependencies]\n", "\n[lints")
dst_deps = block(dst_txt, "[dependencies]\n", "\n[lints")
if src_deps and dst_deps:
    dst_txt = dst_txt.replace(dst_deps, src_deps, 1)
src_lints = block(src_txt, "[lints.clippy]\n", "\n[profile")
dst_lints = block(dst_txt, "[lints.clippy]\n", "\n[profile")
if src_lints and dst_lints:
    dst_txt = dst_txt.replace(dst_lints, src_lints, 1)
if repo_line:
    dst_txt = re.sub(r'^repository = .*', repo_line, dst_txt, count=1, flags=re.M)
open(dest, "w").write(dst_txt)
print("sync-landing: wrote", dest)
if repo_line:
    print("kept", repo_line)
PY
