"""Request-time localization keeps scan evidence canonical (regression for #23595)."""

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_localized_routes_preserve_canonical_scan_and_secret_boundary(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    dashboard = (
        Path(__file__).resolve().parents[2] / "plugins/hermes-achievements/dashboard"
    )
    spec = importlib.util.spec_from_file_location(
        "achievements_localization_test", dashboard / "plugin_api.py"
    )
    assert spec is not None and spec.loader is not None
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, api)
    spec.loader.exec_module(api)
    definitions = api.ACHIEVEMENTS
    visible = next(
        d for d in definitions if not d.get("secret") and "threshold_metric" in d
    )
    secret = next(d for d in definitions if d.get("secret"))
    items = [
        api.display_achievement({
            **d,
            "state": "secret" if d.get("secret") else "unlocked",
            "unlocked": not d.get("secret"),
            "tier": api.TIER_NAMES[0],
            "unlocked_at": i,
        })
        for i, d in enumerate(definitions)
    ]
    snapshot = {
        "achievements": items,
        "sessions": [{"session_id": "sample", "tool_call_count": 200000}],
        "aggregate": {},
        "scan_meta": {"mode": "full"},
        "generated_at": int(time.time()),
        "error": None,
        "unlocked_count": sum(a["unlocked"] for a in items),
        "discovered_count": 0,
        "secret_count": sum(d.get("secret", False) for d in definitions),
        "total_count": len(items),
    }
    setattr(api, "_SNAPSHOT_CACHE", copy.deepcopy(snapshot))
    setattr(api, "_SNAPSHOT_CACHE_AT", snapshot["generated_at"])
    original = copy.deepcopy(api._SNAPSHOT_CACHE)
    app = FastAPI()
    app.include_router(api.router)

    with TestClient(app) as client:
        english_response = client.get("/achievements?locale=en")
        chinese_response = client.get("/achievements?locale=zh-CN")
        assert english_response.status_code == chinese_response.status_code == 200
        english = english_response.json()
        chinese = chinese_response.json()
        by_id = lambda payload: {a["id"]: a for a in payload["achievements"]}
        english_items, chinese_items = by_id(english), by_id(chinese)
        assert (
            chinese_items[visible["id"]]["name"] != english_items[visible["id"]]["name"]
        )
        catalog = json.loads((dashboard / "locales/zh-CN.json").read_text(encoding="utf-8"))
        assert (
            set(english_items) == set(chinese_items) == {d["id"] for d in definitions}
        )
        for definition in definitions:
            item = chinese_items[definition["id"]]
            if definition.get("secret"):
                assert item["name"] == "???" and item["icon"] == "secret"
                assert item["description"] == catalog["._strings"]["secret_hint"]
                assert definition["name"] not in json.dumps(item, ensure_ascii=False)
            else:
                translation = catalog[definition["id"]]
                assert all(
                    item[key] == translation[key]
                    for key in ("name", "description", "category")
                )
                metrics = (
                    [definition["threshold_metric"]]
                    if "threshold_metric" in definition
                    else [r["metric"] for r in definition.get("requirements", [])]
                )
                assert all(
                    catalog["._metrics"][metric] in item["criteria"]
                    for metric in metrics
                )
            assert item["unlocked"] == english_items[definition["id"]]["unlocked"]
        assert (
            chinese_items[secret["id"]]["criteria"]
            != english_items[secret["id"]]["criteria"]
        )
        assert {k: v for k, v in chinese.items() if k != "achievements"} == {
            k: v for k, v in english.items() if k != "achievements"
        }

        for query, header, translated in [
            ("", "zh-CN,zh;q=0.9", True),
            ("", "zh;q=1,zh-CN;q=0,en;q=0.5", False),
            ("", "en;q=1,zh-CN;q=0.5", False),
            ("", "zh-CN;q=bogus,en;q=0.5", False),
            ("?locale=en", "zh-CN", False),
            ("?locale=zh-Hant", "zh-CN", False),
            ("?locale=ja&lang=zh-CN", "zh-CN", False),
            ("?lang=zh-CN", "en", True),
        ]:
            response = client.get(
                "/achievements" + query, headers={"Accept-Language": header}
            )
            assert response.status_code == 200
            expected = chinese_items if translated else english_items
            assert (
                by_id(response.json())[visible["id"]]["name"]
                == expected[visible["id"]]["name"]
            ), (query, header)

        recent_en = client.get("/recent-unlocks?locale=en").json()
        recent_zh = client.get("/recent-unlocks?locale=zh-CN").json()
        assert recent_en and [a["id"] for a in recent_zh] == [
            a["id"] for a in recent_en
        ]
        assert all(a["name"] == catalog[a["id"]]["name"] for a in recent_zh)
        badges_en = client.get("/sessions/sample/badges?locale=en").json()
        badges_zh = client.get("/sessions/sample/badges?locale=zh-CN").json()
        assert badges_en["badges"] and [a["id"] for a in badges_zh["badges"]] == [
            a["id"] for a in badges_en["badges"]
        ]
        assert all(a["name"] == catalog[a["id"]]["name"] for a in badges_zh["badges"])
        assert client.get("/sessions/missing/badges?locale=zh-CN").json() == {
            "session_id": "missing",
            "badges": [],
        }
        assert api._SNAPSHOT_CACHE == original
        assert by_id(client.get("/achievements?locale=en").json()) == english_items

        # Only the expensive scan is replaced; actual rescan handler/cache/persistence run.
        monkeypatch.setattr(
            api, "compute_all", lambda **kwargs: copy.deepcopy(snapshot)
        )
        rescan = client.post("/rescan?locale=zh-CN")
        assert rescan.status_code == 200 and rescan.json()["ok"] is True
        assert by_id(rescan.json()) == chinese_items
        assert api._SNAPSHOT_CACHE == original
        persisted = json.loads(api._data_file("scan_snapshot.json").read_text(encoding="utf-8"))
        assert persisted["achievements"] == original["achievements"]

        # Windows editors may add a BOM; translations must still load from disk.
        locale_dir = tmp_path / "locales"
        locale_dir.mkdir()
        (locale_dir / "zh-CN.json").write_text(
            json.dumps(catalog, ensure_ascii=False), encoding="utf-8-sig"
        )
        monkeypatch.setattr(api, "_LOCALE_DIR", locale_dir)
        api._LOCALE_CACHE.clear()
        assert by_id(client.get("/achievements?locale=zh-CN").json()) == chinese_items
