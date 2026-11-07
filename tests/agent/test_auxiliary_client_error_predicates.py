"""Error-predicate tests for agent.auxiliary_client (moved out of test_auxiliary_client.py)."""


class TestIsTimeoutError:
    """_is_timeout_error distinguishes a full-budget timeout from a fast
    connection drop."""

    def test_timed_out_string(self):
        from agent.auxiliary_client import _is_timeout_error
        assert _is_timeout_error(Exception("Request timed out.")) is True

    def test_timeout_typename(self):
        from agent.auxiliary_client import _is_timeout_error

        class ReadTimeout(Exception):
            pass

        assert _is_timeout_error(ReadTimeout("slow")) is True


class TestIsConnectionError:
    """Tests for _is_connection_error detection."""

    def test_connection_refused(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Connection refused")
        assert _is_connection_error(err) is True



    def test_normal_api_error_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Bad Request: invalid model")
        err.status_code = 400
        assert _is_connection_error(err) is False
