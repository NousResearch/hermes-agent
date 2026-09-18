# Exercise the exported Python hook, not the text of its implementation.
{ pkgs, hook }:
pkgs.runCommand "hermes-dev-python-check" { nativeBuildInputs = [ pkgs.uv ]; } ''
  export HOME="$TMPDIR/home"
  mkdir -p "$HOME" "checkout with spaces"
  cd "checkout with spaces"
  export HERMES_PYTHON_SRC_ROOT="$PWD"
  export VIRTUAL_ENV=/nix/store/inherited-read-only-environment
  ${hook}
  export PATH="$(dirname "$HERMES_PYTHON"):$PATH"
  "$HERMES_PYTHON" - <<'PY'
  import os, sys, sysconfig
  from pathlib import Path
  assert Path(sys.prefix) == Path.cwd() / '.venv', sys.prefix
  assert os.environ['VIRTUAL_ENV'] == sys.prefix
  assert os.access(sysconfig.get_path('purelib'), os.W_OK)
  import pytest  # Nix-provided dependencies remain available without a download.
  PY

  # A local wheel exercises uv's actual install target with network disabled.
  "$HERMES_PYTHON" - <<'PY'
  import zipfile
  files = {
      'hermes_dev_probe.py': 'VALUE = "installed locally"\n',
      'hermes_dev_probe-1.0.dist-info/METADATA': 'Metadata-Version: 2.1\nName: hermes-dev-probe\nVersion: 1.0\n',
      'hermes_dev_probe-1.0.dist-info/WHEEL': 'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
  }
  files['hermes_dev_probe-1.0.dist-info/RECORD'] = "".join(f'{name},,\n' for name in files) + 'hermes_dev_probe-1.0.dist-info/RECORD,,\n'
  with zipfile.ZipFile('hermes_dev_probe-1.0-py3-none-any.whl', 'w') as wheel:
      for name, content in files.items():
          wheel.writestr(name, content)
  PY
  uv pip install --offline --no-index ./hermes_dev_probe-1.0-py3-none-any.whl
  ${hook}
  "$HERMES_PYTHON" - <<'PY'
  import sys
  from pathlib import Path
  import hermes_dev_probe
  assert hermes_dev_probe.VALUE == 'installed locally'
  assert Path(hermes_dev_probe.__file__).is_relative_to(Path(sys.prefix))
  Path('test_local_environment.py').write_text(
      'def test_local_install_is_visible_to_console_scripts():\n'
      '    import hermes_dev_probe\n'
      '    assert hermes_dev_probe.VALUE == "installed locally"\n'
  )
  PY
  pytest --noconftest -q test_local_environment.py
  uv run --active --no-sync python -c 'import hermes_dev_probe'
  touch "$out"
''
