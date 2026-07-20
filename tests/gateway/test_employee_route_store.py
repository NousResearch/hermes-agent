from gateway.config import Platform


def test_employee_route_store_persists_routes_per_platform(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.employee_route_store.get_hermes_home", lambda: tmp_path)

    from gateway.employee_route_store import (
        clear_employee_route_store,
        list_employee_routes,
        set_employee_route,
    )

    set_employee_route(
        Platform.QQ_NAPCAT,
        worker_name="铁柱",
        aliases=["老铁"],
        preloaded_skills=["frontend-design-pro"],
        match_modes=["explicit", "heuristic"],
        action_terms=["打磨"],
        subject_terms=["页面"],
        pain_terms=["粗糙"],
        updated_by="test",
    )
    set_employee_route(
        Platform.WEIXIN,
        worker_name="阿旺",
        aliases=["旺财"],
        preloaded_skills=["ops-skill"],
        updated_by="test",
    )

    qq_routes = list_employee_routes(Platform.QQ_NAPCAT)
    weixin_routes = list_employee_routes(Platform.WEIXIN)

    assert qq_routes == [
        {
            "worker_name": "铁柱",
            "aliases": ["老铁"],
            "preloaded_skills": ["frontend-design-pro"],
            "match_modes": ("explicit", "heuristic"),
            "action_terms": ("打磨",),
            "subject_terms": ("页面",),
            "pain_terms": ("粗糙",),
            "updated_at": qq_routes[0]["updated_at"],
            "updated_by": "test",
            "enabled": True,
        }
    ]
    assert weixin_routes == [
        {
            "worker_name": "阿旺",
            "aliases": ["旺财"],
            "preloaded_skills": ["ops-skill"],
            "match_modes": ("explicit",),
            "action_terms": (),
            "subject_terms": (),
            "pain_terms": (),
            "updated_at": weixin_routes[0]["updated_at"],
            "updated_by": "test",
            "enabled": True,
        }
    ]

    cleared = clear_employee_route_store(Platform.QQ_NAPCAT, "铁柱", updated_by="cleanup")
    assert cleared["worker_name"] == "铁柱"
    assert list_employee_routes(Platform.QQ_NAPCAT) == []
    assert list_employee_routes(Platform.WEIXIN)[0]["worker_name"] == "阿旺"
