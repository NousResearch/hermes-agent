"""The truststore handshake-window false positive on AWS endpoints (#132905).

truststore's macOS/Windows backends flip the shared SSLContext to
``check_hostname=False`` / ``verify_mode=CERT_NONE`` for the duration of each
``wrap_socket`` and restore it after; the lock guards entry, not other
readers. urllib3 decides ``is_verified`` by reading ``context.verify_mode``
right after the handshake, so with one shared context — exactly what botocore
keeps per Bedrock client — a concurrent handshake makes it read the transient
``CERT_NONE`` and warn about a request truststore itself verified through the
OS store.

These contracts pin the narrow mute ``install_truststore()`` installs: only
that warning, only for ``*.amazonaws.com``, only under the platform verifier.
"""

import builtins
import sys
import types
import warnings

import agent.ssl_verify
from urllib3.exceptions import InsecureRequestWarning

AWS_WARNING = (
    "Unverified HTTPS request is being made to host "
    "'bedrock-runtime.eu-west-1.amazonaws.com'. Adding certificate verification is strongly advised."
)
AWS_CN_WARNING = (
    "Unverified HTTPS request is being made to host "
    "'bedrock-runtime.cn-north-1.amazonaws.com.cn'. Adding certificate verification is strongly advised."
)
CORP_WARNING = (
    "Unverified HTTPS request is being made to host "
    "'internal.corp.local'. Adding certificate verification is strongly advised."
)


def _capture_shown():
    shown: list = []
    original = warnings.showwarning

    def record(*args, **kwargs):
        shown.append(args)

    warnings.showwarning = record
    return shown, original


def test_truststore_install_mutes_only_the_amazonaws_handshake_race(monkeypatch):
    """Hermes never disables verification on its boto3 clients, so under the
    platform verifier an ``*.amazonaws.com`` unverified-request warning is
    always the race: muted. Every other host, message, or category keeps its
    default disposition."""
    fake = types.ModuleType("truststore")
    fake.inject_into_ssl = lambda: None
    monkeypatch.setitem(sys.modules, "truststore", fake)
    monkeypatch.setattr(agent.ssl_verify, "_installed", None)

    saved_filters = warnings.filters[:]
    shown, original_showwarning = _capture_shown()
    try:
        assert agent.ssl_verify.install_truststore() is True

        def muted(message, category):
            before = len(shown)
            warnings.warn_explicit(
                message, category, "handshake_race.py", 1, registry={}
            )
            return len(shown) == before

        assert muted(AWS_WARNING, InsecureRequestWarning), (
            "the truststore race false positive should be muted"
        )
        assert muted(AWS_CN_WARNING, InsecureRequestWarning), (
            "the China-region endpoint form should be muted too"
        )
        assert not muted(CORP_WARNING, InsecureRequestWarning), (
            "a non-AWS host that is really unverified must still warn"
        )
        assert not muted(AWS_WARNING, UserWarning), (
            "the same message in another category must still warn"
        )
        assert not muted("Some other urllib3 grievance", InsecureRequestWarning), (
            "other messages in the category must still warn"
        )
    finally:
        warnings.showwarning = original_showwarning
        warnings.filters[:] = saved_filters


def test_without_the_platform_verifier_the_insecure_warning_survives(monkeypatch):
    """The mute is conditional on truststore being in force: with the
    verifier absent the same AWS warning must surface, because then it can
    be a genuine unverified request."""
    real_import = builtins.__import__

    def no_truststore(name, *args, **kwargs):
        if name == "truststore":
            raise ImportError("unavailable fixture")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_truststore)
    monkeypatch.delitem(sys.modules, "truststore", raising=False)
    monkeypatch.setattr(agent.ssl_verify, "_installed", None)

    saved_filters = warnings.filters[:]
    shown, original_showwarning = _capture_shown()
    try:
        assert agent.ssl_verify.install_truststore() is False
        warnings.warn_explicit(
            AWS_WARNING, InsecureRequestWarning, "handshake_race.py", 1, registry={}
        )
        assert len(shown) == 1, "without the platform verifier the warning must survive"
    finally:
        warnings.showwarning = original_showwarning
        warnings.filters[:] = saved_filters
