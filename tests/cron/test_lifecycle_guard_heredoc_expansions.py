"""Quoted data heredocs retain executable headers without walking data as scripts (#152447)."""

import pytest

from cron.lifecycle_guard import scan_gateway_lifecycle
from tools.shell_heredoc import strip_inert_heredoc_bodies


@pytest.mark.parametrize(
    "opener",
    [
        'cat > "$(cygpath -m "$TMPDIR")/x.py" <<\'PY\'',
        'cat <<\'PY\' > "$(printf "%s" "$TMPDIR")/x.py"',
        'cat > "$(printf "%s" "$(pwd)")/x.py" <<\'PY\'',
        'cat > "`pwd`/x.py" <<\'PY\'',
        "cat > $(pwd)/x.py <<'PY'",
        "V=$(uname -s); cat > ./x.py <<'PY'",
        "V=$(printf x; printf y); python3 - <<'PY'",
        "cat > $(printf '#') <<'PY'",
        'cat > $(printf "#") <<\'PY\'',
        "cat > $(printf prefix#suffix) <<'PY'",
        "cat > `printf '#'` <<'PY'",
    ],
)
def test_header_expansions_do_not_turn_quoted_data_into_executed_scripts(tmp_path, opener):
    data = tmp_path / "train.parquet"
    data.write_bytes(b"\0" * (2 * 1024 * 1024))
    command = f"{opener}\nimport pandas as pd\ntrain = pd.read_parquet('{data}')\nPY"

    assert scan_gateway_lifecycle(command, cwd=str(tmp_path)) == (False, None)
    # Header substitutions still execute; only the body disappears from the executed view.
    masked = strip_inert_heredoc_bodies(command)
    assert masked.startswith(opener + "\n")
    assert masked.count("\n") == command.count("\n")


@pytest.mark.parametrize(
    "opener",
    [
        'cat > "$(hermes gateway restart; pwd)/x.py" <<\'PY\'',
        'cat > "`hermes gateway restart; pwd`/x.py" <<\'PY\'',
        "V=$(hermes gateway restart); cat > ./x.py <<'PY'",
    ],
)
def test_executable_header_is_still_scanned_when_the_body_is_data(tmp_path, opener):
    command = f"{opener}\nprint('data')\nPY"
    assert strip_inert_heredoc_bodies(command) == opener + "\n\n"
    assert scan_gateway_lifecycle(command, cwd=str(tmp_path)) == (True, None)


@pytest.mark.parametrize(("expansion", "closer"), [("$(", ")"), ("<(", ")"), (">(", ")"), ("`", "`")])
def test_commented_closer_does_not_hide_executable_substitution_lines(tmp_path, expansion, closer):
    command = f"cat > {expansion}printf sink # {closer} <<':'\nhermes gateway restart\n:\n{closer} </dev/null"
    assert strip_inert_heredoc_bodies(command) == command
    assert scan_gateway_lifecycle(command, cwd=str(tmp_path)) == (True, None)
