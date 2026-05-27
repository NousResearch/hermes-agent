from run_agent import _is_nonretryable_local_validation_error


def test_nonetype_not_iterable_is_retryable_provider_shape_error():
    error = TypeError("'NoneType' object is not iterable")

    assert _is_nonretryable_local_validation_error(error) is False


def test_plain_type_error_remains_nonretryable_local_validation_error():
    error = TypeError("tools must be a list")

    assert _is_nonretryable_local_validation_error(error) is True


def test_value_error_remains_nonretryable_local_validation_error():
    error = ValueError("invalid local request")

    assert _is_nonretryable_local_validation_error(error) is True
