"""The session.create coherence gate must not refuse an id the provider's own listing serves (#138969).

``static_model_provider_conflict`` checks membership in the curated ``_PROVIDER_MODELS`` catalog.
``hy4-preview`` is served live by the OpenCode Go relay and the ``/model`` picker (which merges the
live listing) offers it — but until the curated tuple caught up, the slug was only listed by the
tencent-tokenhub/tencent-tokenplan catalogs, so the gate classified it as a cross-vendor conflict
and killed the session with "Model `hy4-preview` is not served by provider `opencode-go`".
"""

from __future__ import annotations

from hermes_cli import models as m
from hermes_cli.models_validate import static_model_provider_conflict


def test_live_served_slug_of_a_new_family_passes_the_gate():
    # Red before the curated tuple gained "hy4-preview": family head "hy4" matched no opencode-go
    # family and the tencent catalogs claimed the slug.
    assert static_model_provider_conflict("hy4-preview", "opencode-go") is None


def test_picker_and_gate_agree_on_the_served_slug():
    # Whatever the gate refuses, the picker must not offer (#138969's invariant).
    assert "hy4-preview" in m.provider_model_ids("opencode-go")
    assert static_model_provider_conflict("hy4-preview", "opencode-go") is None


def test_foreign_catalog_slug_still_refused():
    # "big-pickle" is a Zen-relay exclusive: not in the Go catalog's families, claimed by another
    # vendor's curated catalog — the gate must keep refusing it on opencode-go.
    verdict = static_model_provider_conflict("big-pickle", "opencode-go")
    assert verdict is not None and "not served by provider `opencode-go`" in verdict["message"]
