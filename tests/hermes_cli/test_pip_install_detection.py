from unittest.mock import patch

import pytest










def test_code_scoped_stamp_wins_over_home_stamp(tmp_path):
    """The stamp next to the running code is authoritative over $HERMES_HOME.

    Models a host git install whose $HERMES_HOME is shared with (and stamped
    'docker' by) a co-located container. The code-scoped stamp must win so the
    host install is correctly identified as 'git' and 'hermes update' works.
    """
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    (code / ".install_method").write_text("git\n")
    (home / ".install_method").write_text("docker\n")  # container contamination
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=code) == "git"


def test_container_without_stamp_is_not_docker(tmp_path):
    """An unstamped install in a generic container must NOT be flagged as docker.

    Regression for issue #34397. The two supported installs both stamp
    ``.install_method`` (the curl installer -> ``git``, covered by
    ``test_stamp_file_takes_precedence``; the published image -> ``docker``),
    so neither hits this path. An unsupported manual install dropped into a
    container has no stamp and was wrongly classified as the published Docker
    image, so ``hermes update`` refused to run. With a ``.git`` checkout it
    must resolve to ``git``.
    """
    (tmp_path / ".git").mkdir()
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path), \
         patch("hermes_constants.is_container", return_value=True):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "git"






