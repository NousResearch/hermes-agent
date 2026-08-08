#!/usr/bin/env python3
import json
import sys
from pathlib import Path

HEAVY = set('@#$%&_~')
LIGHT = set("Ili;:,. '`")


def load_frames(path: Path):
    data = path.read_text().splitlines()
    state = None
    frames = []
    for line in data[1:]:
        obj = json.loads(line)
        if 'state' in obj:
            state = obj['state']
            continue
        if 'data' in obj:
            frames.append((state, obj['data']))
    return frames


def score(text: str):
    chars = [c for c in text if c != "\n"]
    nonspace = [c for c in chars if c != ' ']
    total = max(1, len(nonspace))
    uniq = len(set(nonspace))
    heavy = sum(c in HEAVY for c in nonspace) / total
    light = sum(c in LIGHT for c in nonspace) / total
    return uniq, heavy, light


def main():
    if len(sys.argv) != 2:
        print('usage: diagnose-glyph-quality.py <file.eikon>')
        raise SystemExit(1)
    p = Path(sys.argv[1])
    frames = load_frames(p)
    if not frames:
        print('no frames found')
        raise SystemExit(2)
    sample = frames[0][1]
    uniq, heavy, light = score(sample)
    print(json.dumps({
        'file': str(p),
        'frames': len(frames),
        'sample_unique_glyphs': uniq,
        'sample_heavy_ratio': round(heavy, 4),
        'sample_light_ratio': round(light, 4),
    }, indent=2))


if __name__ == '__main__':
    main()
