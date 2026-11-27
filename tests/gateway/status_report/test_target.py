"""The switch: ``GATEWAY_STATUS_REPORT_URL`` in the launch environment, with its credential split off."""

from __future__ import annotations

import logging

import pytest

from gateway.status_report.target import STATUS_REPORT_URL_ENV, report_target_from_env
from tests.gateway.status_report.support import REPORT_URL, new_credential, stamped_url


@pytest.mark.parametrize("environ", [{}, {STATUS_REPORT_URL_ENV: ""}, {STATUS_REPORT_URL_ENV: "   "}])
def test_no_url_means_no_target_and_says_nothing(environ, caplog):
    caplog.set_level(logging.DEBUG)

    assert report_target_from_env(environ) is None

    assert [record for record in caplog.records if record.name.startswith("gateway.status_report")] == []


def test_the_credential_is_split_from_the_url():
    credential = new_credential()

    target = report_target_from_env({STATUS_REPORT_URL_ENV: f"{REPORT_URL}?zone=eu&t={credential}&shard=2"})

    assert target is not None
    assert target.credential == credential
    assert target.url == f"{REPORT_URL}?zone=eu&shard=2"
    assert credential not in repr(target) and "status.example" not in repr(target)


def test_a_url_with_only_the_credential_keeps_no_query():
    credential = new_credential()

    target = report_target_from_env({STATUS_REPORT_URL_ENV: stamped_url(credential)})

    assert target is not None and target.url == REPORT_URL


@pytest.mark.parametrize(
    "shape",
    [
        "http://status.example/report?t={c}",  # not https
        "status.example/report?t={c}",  # not absolute
        "https:///report?t={c}",  # no host
        "https://status.example/report",  # no credential
        "https://status.example/report?t=",  # empty credential
        "https://status.example/report?t={c}&t={c}",  # two credentials
        "https://user:{c}@status.example/report?t={c}",  # a second credential in the address
        "https://status.example:port/report?t={c}",  # not a port
        "https://status.example/report?t={c}%0d%0aX-Injected:%201",  # would break out of the header
        "https://status.example/report?t={c}%20{c}",  # not one token
    ],
)
def test_a_malformed_url_is_ignored_and_the_warning_names_only_the_key(shape, caplog):
    credential = new_credential()
    caplog.set_level(logging.DEBUG)

    assert report_target_from_env({STATUS_REPORT_URL_ENV: shape.format(c=credential)}) is None

    assert STATUS_REPORT_URL_ENV in caplog.text
    assert credential not in caplog.text and "status.example" not in caplog.text
