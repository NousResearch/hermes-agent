from __future__ import annotations

import json
from unittest import mock

import pytest

from hermes_cli.cron import _cron_api


def test_cron_api_normal_dict_payload():
    with mock.patch(
        "tools.cronjob_tools.cronjob",
        return_value=json.dumps({"success": True, "jobs": []}),
    ):
        result = _cron_api(action="list")
    assert result == {"success": True, "jobs": []}


def test_cron_api_raw_json_string_payload():
    with mock.patch("tools.cronjob_tools.cronjob", return_value='{"raw": "value"}'):
        result = _cron_api(action="list")
    assert result == {"raw": "value"}


def test_cron_api_non_dict_json_payload():
    with mock.patch("tools.cronjob_tools.cronjob", return_value="[]"):
        result = _cron_api(action="list")
    assert result == {"raw": []}


def test_cron_api_non_json_payload():
    with mock.patch("tools.cronjob_tools.cronjob", return_value="not-json"):
        result = _cron_api(action="list")
    assert result == {"raw": "not-json"}


def test_cron_api_none_payload():
    with mock.patch("tools.cronjob_tools.cronjob", return_value=None):
        result = _cron_api(action="list")
    assert result == {"raw": None}


def test_cron_api_typeerror_payload():
    with mock.patch("tools.cronjob_tools.cronjob", side_effect=TypeError("boom")):
        with pytest.raises(TypeError):
            _cron_api(action="list")
