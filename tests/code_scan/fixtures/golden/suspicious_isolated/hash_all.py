import hashlib
import json
import os
import sys


def compute_hashes(directory: str) -> dict:
    """Compute SHA-256 hashes for every file in a directory."""
    result = {}
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "rb") as f:
                    h = hashlib.sha256(f.read()).hexdigest()
                result[fpath] = h
            except OSError:
                result[fpath] = "ERROR"
    return result


def main() -> int:
    """Entry point."""
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    hashes = compute_hashes(target)
    print(json.dumps(hashes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
