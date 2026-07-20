from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.employee_routes import get_employee_routes


def test_get_employee_routes_returns_empty_without_platform_override():
    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {},
                }
            }
        }
    )

    assert get_employee_routes(config, platform=Platform.QQ_NAPCAT) == []


def test_get_employee_routes_prefers_platform_override():
    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {
                        "employee_routes": [
                            {
                                "worker_name": "阿旺",
                                "aliases": ["旺财"],
                                "preloaded_skills": ["ops-skill"],
                                "action_terms": ["排查"],
                                "subject_terms": ["服务器"],
                                "pain_terms": ["炸了"],
                            }
                        ]
                    },
                }
            }
        }
    )

    assert get_employee_routes(config, platform=Platform.QQ_NAPCAT) == [
        {
            "worker_name": "阿旺",
            "aliases": ["旺财"],
            "preloaded_skills": ["ops-skill"],
            "match_modes": ("explicit",),
            "action_terms": ("排查",),
            "subject_terms": ("服务器",),
            "pain_terms": ("炸了",),
        }
    ]


def test_get_employee_routes_filters_disabled_and_invalid_entries():
    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {
                        "employee_routes": [
                            {"worker_name": "禁用员工", "enabled": False},
                            {"aliases": ["没名字"]},
                            "not-a-dict",
                            {
                                "worker_name": "阿旺",
                                "aliases": "旺财",
                                "skills": "ops-skill",
                                "action_terms": ["排查", ""],
                                "subject_terms": "服务器",
                                "pain_terms": None,
                            },
                        ]
                    },
                }
            }
        }
    )

    assert get_employee_routes(config, platform=Platform.QQ_NAPCAT) == [
        {
            "worker_name": "阿旺",
            "aliases": ["旺财"],
            "preloaded_skills": ["ops-skill"],
            "match_modes": ("explicit",),
            "action_terms": ("排查",),
            "subject_terms": ("服务器",),
            "pain_terms": (),
        }
    ]


def test_get_employee_routes_degrades_safely_on_malformed_override():
    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {"employee_routes": {"worker_name": "阿旺"}},
                }
            }
        }
    )

    assert get_employee_routes(config, platform=Platform.QQ_NAPCAT) == []


def test_load_gateway_config_reads_qq_napcat_employee_routes(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(
        """
qq_napcat:
  employee_routes:
    - worker_name: 阿旺
      aliases: [旺财]
      preloaded_skills: [ops-skill]
      action_terms: [排查]
      subject_terms: [服务器]
      pain_terms: [炸了]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_gateway_config()
    routes = get_employee_routes(config, platform=Platform.QQ_NAPCAT)

    assert routes == [
        {
            "worker_name": "阿旺",
            "aliases": ["旺财"],
            "preloaded_skills": ["ops-skill"],
            "match_modes": ("explicit",),
            "action_terms": ("排查",),
            "subject_terms": ("服务器",),
            "pain_terms": ("炸了",),
        }
    ]


def test_get_employee_routes_supports_explicit_and_heuristic_modes():
    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {
                        "employee_routes": [
                            {
                                "worker_name": "铁柱",
                                "preloaded_skills": ["frontend-design-pro"],
                                "match_modes": ["explicit", "heuristic"],
                                "routing_hints": {
                                    "action_terms": ["打磨"],
                                    "subject_terms": ["页面"],
                                    "pain_terms": ["粗糙"],
                                },
                            }
                        ]
                    },
                }
            }
        }
    )

    assert get_employee_routes(config, platform=Platform.QQ_NAPCAT) == [
        {
            "worker_name": "铁柱",
            "aliases": [],
            "preloaded_skills": ["frontend-design-pro"],
            "match_modes": ("explicit", "heuristic"),
            "action_terms": ("打磨",),
            "subject_terms": ("页面",),
            "pain_terms": ("粗糙",),
        }
    ]


def test_get_employee_routes_merges_dynamic_store_routes(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.employee_route_store.get_hermes_home", lambda: tmp_path)

    from gateway.employee_route_store import set_employee_route

    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {
                        "employee_routes": [
                            {
                                "worker_name": "阿旺",
                                "preloaded_skills": ["ops-skill"],
                            }
                        ]
                    },
                }
            }
        }
    )

    set_employee_route(
        Platform.QQ_NAPCAT,
        worker_name="铁柱",
        aliases=["老铁"],
        preloaded_skills=["frontend-design-pro"],
        match_modes=["explicit"],
        updated_by="test",
    )

    assert get_employee_routes(config, platform=Platform.QQ_NAPCAT) == [
        {
            "worker_name": "阿旺",
            "aliases": [],
            "preloaded_skills": ["ops-skill"],
            "match_modes": ("explicit",),
            "action_terms": (),
            "subject_terms": (),
            "pain_terms": (),
        },
        {
            "worker_name": "铁柱",
            "aliases": ["老铁"],
            "preloaded_skills": ["frontend-design-pro"],
            "match_modes": ("explicit",),
            "action_terms": (),
            "subject_terms": (),
            "pain_terms": (),
        },
    ]


def test_get_employee_routes_dynamic_store_overrides_static_route_by_worker_name(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.employee_route_store.get_hermes_home", lambda: tmp_path)

    from gateway.employee_route_store import set_employee_route

    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {
                        "employee_routes": [
                            {
                                "worker_name": "铁柱",
                                "aliases": ["旧铁柱"],
                                "preloaded_skills": ["old-skill"],
                            }
                        ]
                    },
                }
            }
        }
    )

    set_employee_route(
        Platform.QQ_NAPCAT,
        worker_name="铁柱",
        aliases=["老铁"],
        preloaded_skills=["frontend-design-pro"],
        match_modes=["explicit", "heuristic"],
        action_terms=["打磨"],
        updated_by="test",
    )

    assert get_employee_routes(config, platform=Platform.QQ_NAPCAT) == [
        {
            "worker_name": "铁柱",
            "aliases": ["老铁"],
            "preloaded_skills": ["frontend-design-pro"],
            "match_modes": ("explicit", "heuristic"),
            "action_terms": ("打磨",),
            "subject_terms": (),
            "pain_terms": (),
        }
    ]
