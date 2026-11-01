"""Quoted or escaped control characters are shell words, not operators.

POSIX shlex dequotes ``\\(`` and ``'('`` to the same token text as a real ``(``, so the guard split
the segment there and put the next argument in command position: ``magick \\( in.png ...`` read a
>1 MiB image as an executed script and failed closed.
"""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard


def _big_file(tmp_path):
    path = tmp_path / "in.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * (2 * 1024 * 1024))
    return path


@pytest.mark.parametrize(
    "template",
    [
        "magick \\( {big} -negate \\) {big} out.png",
        "magick '(' {big} -negate ')' {big} out.png",
        'magick "(" {big} -negate ")" {big} out.png',
        "tr '|' {big} < /dev/null",
        "find . -exec true {{}} \\; {big}",
    ],
)
def test_quoted_or_escaped_operator_does_not_execute_next_argument(tmp_path, template):
    big = _big_file(tmp_path)
    assert guard(template.format(big=big), cwd=str(tmp_path)) is False


@pytest.mark.parametrize(
    "template",
    [
        "( {script} )",
        "true | {script}",
        "true; {script}",
        "echo '(' && {script}",
        "magick \\( in.png \\) out.png; {script}",
        "true # it's\ntrue; {script}",
        'true # "\ntrue && {script}',
        "eval true \\; sh {script}",
        "eval true ';' {script}",
        "! eval true \\; sh {script}",
        "watch -n 1 true \\; sh {script}",
        "busybox watch -t -n 1 true \\; sh {script}",
        "ssh localhost true \\; sh {script}",
        "parallel --plain true \\; sh {script} ::: x",
    ],
)
def test_real_operator_still_executes_next_argument(tmp_path, template):
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    assert guard(template.format(script=script), cwd=str(tmp_path)) is True


def _chain(tmp_path, prefix, length):
    """`<prefix>0.sh` runs `<prefix>1.sh` ... the last one restarts the gateway."""
    scripts = [tmp_path / f"{prefix}{index}.sh" for index in range(length)]
    for current, following in zip(scripts, scripts[1:]):
        current.write_text(f"#!/bin/sh\nsh {following}\n", encoding="utf-8")
    scripts[-1].write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    return scripts[0]


def test_readable_chain_behind_escaped_operator_still_fails_closed_at_the_depth_limit(tmp_path):
    first = _chain(tmp_path, "s", 9)
    assert guard(f"eval true \\; sh {first}", cwd=str(tmp_path)) is True


def test_skipped_mention_does_not_hide_a_later_real_execution(tmp_path):
    first = _chain(tmp_path, "b", 8)
    mentioner = tmp_path / "a.sh"
    mentioner.write_text(f"#!/bin/sh\necho ';' {first}\n", encoding="utf-8")
    assert guard(f"sh {mentioner}; sh {first}", cwd=str(tmp_path)) is True


def test_heredoc_mention_does_not_hide_a_readable_chain_behind_an_escaped_operator(tmp_path):
    first = _chain(tmp_path, "c", 8)
    wrapper = tmp_path / "w.sh"
    wrapper.write_text(
        f"#!/bin/sh\npython3 - <<'PY'\nimport os\nos.system('{first}')\nPY\n", encoding="utf-8"
    )
    assert guard(f"eval true \\; {first}; sh {wrapper}", cwd=str(tmp_path)) is True


def test_walk_budget_exhaustion_still_fails_closed_for_soft_candidates(tmp_path):
    """Both files are below the per-script cap and the line budget; together they exceed the walk
    byte budget, so the soft `restart.sh` cannot be read in full and must still fail closed."""
    pad = tmp_path / "pad.sh"
    pad.write_text("#!/bin/sh\n" + ("#" + "x" * 998 + "\n") * 800, encoding="utf-8")
    restart = tmp_path / "restart.sh"
    restart.write_text(
        "#!/bin/sh\n" + ("#" + "y" * 998 + "\n") * 250 + "hermes gateway restart\n", encoding="utf-8"
    )
    assert guard(f"sh {pad}", cwd=str(tmp_path)) is False
    assert guard(f"sh {pad}; eval true \\; sh {restart}", cwd=str(tmp_path)) is True


def test_repeated_oversized_soft_file_is_read_once(tmp_path):
    _big_file(tmp_path)
    command = "magick " + " ".join(["\\( ./in.png \\)"] * 1100) + " out.png"
    assert guard(command, cwd=str(tmp_path)) is False
