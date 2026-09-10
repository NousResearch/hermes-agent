import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
INIT_PATH = PLUGIN_DIR / '__init__.py'
CORE_PATH = PLUGIN_DIR / 'src' / 'hashline_core.py'
TARGET = Path(tempfile.gettempdir()) / 'hashline-anchored-e2e.txt'


def _load_plugin():
    spec = importlib.util.spec_from_file_location('hashline_guard_plugin', INIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_core():
    spec = importlib.util.spec_from_file_location('hashline_core', CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def main() -> int:
    plugin = _load_plugin()
    core = _load_core()
    failures = []

    # Step 1 create temp file (write bytes to avoid Windows newline translation)
    # Anchors are NON-adjacent (line 2 and line 5) so their context windows are
    # independent: patching one must not change the other's hashline.
    original = 'alpha\nbeta\ngamma\ndelta\nbeta\nepsilon\n'
    TARGET.write_bytes(original.encode('utf-8'))
    assert _bytes(TARGET) == original.encode('utf-8'), 'step1 write failed'

    # Step 2 hashline_compute on old_string='beta', assert count=2 and hashes differ
    text = TARGET.read_text(encoding='utf-8')
    matches = core.find_all(text, 'beta')
    if len(matches) != 2:
        failures.append(f'step2 expected 2 matches, got {len(matches)}')
    h0 = core.compute_hashline(text, 'beta', 0)
    h1 = core.compute_hashline(text, 'beta', 1)
    if h0 == h1:
        failures.append(f'step2 hashes should differ: {h0} == {h1}')

    # Step 3 pin occurrence 1 (first beta, line 2) -> replace with B1
    before_step3 = _bytes(TARGET)
    res3 = json.loads(plugin.handle_anchored_patch({
        'path': str(TARGET),
        'old_string': 'beta',
        'new_string': 'B1',
        'expected_hashline': h0,
        'window': 2,
    }))
    if not res3.get('applied'):
        failures.append(f'step3 not applied: {res3}')
    if res3.get('occurrence') != 0:
        failures.append(f"step3 expected occurrence 0, got {res3.get('occurrence')}")
    if _bytes(TARGET) == before_step3:
        failures.append('step3 file bytes unchanged after successful patch')
    after_step3 = TARGET.read_text(encoding='utf-8')
    if after_step3 != 'alpha\nB1\ngamma\ndelta\nbeta\nepsilon\n':
        failures.append(f'step3 wrong content: {after_step3!r}')

    # Step 4 recompute h1 on the current file state because the original
    # occurrence 1 patch changed surrounding context/line numbers.
    h1 = core.compute_hashline(TARGET.read_text(encoding='utf-8'), 'beta', 0)
    before_step4 = _bytes(TARGET)
    res4 = json.loads(plugin.handle_anchored_patch({
        'path': str(TARGET),
        'old_string': 'beta',
        'new_string': 'B2',
        'expected_hashline': h1,
        'window': 2,
    }))
    if not res4.get('applied'):
        failures.append(f'step4 not applied: {res4}')
    if res4.get('occurrence') != 0:
        failures.append(f"step4 expected occurrence 0 (only beta left), got {res4.get('occurrence')}")
    if _bytes(TARGET) == before_step4:
        failures.append('step4 file bytes unchanged after successful patch')
    after_step4 = TARGET.read_text(encoding='utf-8')
    if after_step4 != 'alpha\nB1\ngamma\ndelta\nB2\nepsilon\n':
        failures.append(f'step4 wrong content: {after_step4!r}')

    # Step 5 wrong hashline -> block, file unchanged
    before_step5 = _bytes(TARGET)
    res5 = json.loads(plugin.handle_anchored_patch({
        'path': str(TARGET),
        'old_string': 'beta',
        'new_string': 'B2',
        'expected_hashline': '0' * 64,
        'window': 2,
    }))
    if res5.get('applied'):
        failures.append('step5 should have blocked on wrong hashline')
    if _bytes(TARGET) != before_step5:
        failures.append('step5 file bytes changed on block')

    # Step 6 CRLF normalization -> still works
    TARGET.write_bytes(b'alpha\r\nbeta\r\ngamma\r\ndelta\r\nbeta\r\nepsilon\r\n')
    text_crlf = TARGET.read_text(encoding='utf-8')
    matches_crlf = core.find_all(text_crlf, 'beta')
    if len(matches_crlf) != 2:
        failures.append(f'step6 expected 2 matches after CRLF, got {len(matches_crlf)}')
    h0_crlf = core.compute_hashline(text_crlf, 'beta', 0)
    h1_crlf = core.compute_hashline(text_crlf, 'beta', 1)
    before_step6 = _bytes(TARGET)
    res6 = json.loads(plugin.handle_anchored_patch({
        'path': str(TARGET),
        'old_string': 'beta',
        'new_string': 'B1',
        'expected_hashline': h0_crlf,
        'window': 2,
    }))
    if not res6.get('applied'):
        failures.append(f'step6 not applied on CRLF-normalized file: {res6}')
    if _bytes(TARGET) == before_step6:
        failures.append('step6 file bytes unchanged after successful patch')
    # Byte-exact check: the patch must splice at the RAW offsets, preserving
    # CRLF endings (regression for canonical-vs-raw offset corruption where a
    # CRLF file's anchor became 'B1a' with a stray '\r').
    expected_bytes6 = b'alpha\r\nB1\r\ngamma\r\ndelta\r\nbeta\r\nepsilon\r\n'
    if _bytes(TARGET) != expected_bytes6:
        failures.append(f'step6 CRLF bytes wrong: {_bytes(TARGET)!r} (expected {expected_bytes6!r})')

    if failures:
        print('FAILURES:')
        for f in failures:
            print(' -', f)
        return 1
    print('ALL ASSERTIONS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
