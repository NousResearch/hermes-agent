from __future__ import annotations

import json

from agent.pet import manifest


def _payload(slug: str) -> dict:
    return {
        "pets": [
            {
                "slug": slug,
                "displayName": slug.title(),
                "spritesheetUrl": f"https://assets.petdex.dev/{slug}.webp",
            }
        ]
    }


def test_local_manifest_does_not_pollute_network_cache(tmp_path, monkeypatch) -> None:
    local = tmp_path / "manifest.json"
    local.write_text(json.dumps(_payload("local-pet")), encoding="utf-8")

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return _payload("network-pet")

    calls: list[dict] = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr("httpx.get", fake_get)
    manifest.clear_cache()

    assert manifest.fetch_manifest(local_path=str(local))[0].slug == "local-pet"
    assert manifest.fetch_manifest()[0].slug == "network-pet"
    assert len(calls) == 1


def test_find_entry_forwards_insecure_manifest_setting(monkeypatch) -> None:
    captured: dict = {}

    def fake_fetch_manifest(**kwargs):
        captured.update(kwargs)
        return [manifest.ManifestEntry.from_dict(_payload("boba")["pets"][0])]

    monkeypatch.setattr(manifest, "fetch_manifest", fake_fetch_manifest)

    assert manifest.find_entry("boba", verify=False).slug == "boba"
    assert captured["verify"] is False
