from unittest.mock import patch


def test_check_via_pypi_detects_update():
    """_check_via_pypi returns 1 when PyPI has newer version."""
    from hermes_cli.banner import _check_via_pypi
    with patch("hermes_cli.banner.VERSION", "0.12.0"):
        with patch("hermes_cli.banner._fetch_pypi_latest", return_value="0.13.0"):
            result = _check_via_pypi()
            assert result == 1


def test_check_via_pypi_up_to_date():
    """_check_via_pypi returns 0 when versions match."""
    from hermes_cli.banner import _check_via_pypi
    with patch("hermes_cli.banner.VERSION", "0.13.0"):
        with patch("hermes_cli.banner._fetch_pypi_latest", return_value="0.13.0"):
            result = _check_via_pypi()
            assert result == 0


def test_check_via_pypi_network_failure():
    """_check_via_pypi returns None on network error."""
    from hermes_cli.banner import _check_via_pypi
    with patch("hermes_cli.banner._fetch_pypi_latest", return_value=None):
        result = _check_via_pypi()
        assert result is None


def test_version_tuple_comparison():
    """Version comparison works with multi-segment versions."""
    from hermes_cli.banner import _version_tuple
    assert _version_tuple("0.13.0") > _version_tuple("0.12.0")
    assert _version_tuple("0.13.0") == _version_tuple("0.13.0")
    assert _version_tuple("1.0.0") > _version_tuple("0.99.99")
