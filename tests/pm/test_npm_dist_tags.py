"""npm metadata keeps escaped names, caller headers and retry policy."""
import json
from pm.packages import Npm
from pm.update import npm_dist_tags, resolve_package
from tests.pm.test_update_request_reuse import upstream
from tests.pm._range_server import RangeHandler, dl_server


def test_tag_endpoint_drives_package_updates_and_preserves_escaped_names(upstream, monkeypatch):
    from hermes_cli import urllib_security
    calls, failures = upstream
    tags = {
        "/-/package/npm/dist-tags": {"latest": "2.3.4", "next": "3.0.0-beta.1"},
        "/-/package/@scope%2Ftool/dist-tags": {},
    }
    RangeHandler.payloads.update({path: json.dumps(value).encode() for path, value in tags.items()})
    failures["/-/package/npm/dist-tags"] = [503]
    original = urllib_security.open_credentialed_url
    def open_registry(request, **kwargs):
        assert request.full_url.startswith("https://registry.npmjs.org/")
        assert request.get_header("User-agent") == "hermes-pm"
        return original(request, **kwargs)
    monkeypatch.setattr(urllib_security, "open_credentialed_url", open_registry)
    monkeypatch.setenv("GH_TOKEN", "not-for-npm")
    monkeypatch.setenv("HF_TOKEN", "not-for-npm-either")
    decision = resolve_package(Npm(), ["linux-x64"], locked="1.0.0")
    assert decision.version == "2.3.4" and decision.changed
    assert npm_dist_tags("@scope%2Ftool") == {}
    assert calls == [("/-/package/npm/dist-tags", None), ("/-/package/npm/dist-tags", None),
                     ("/-/package/@scope%2Ftool/dist-tags", None)]
