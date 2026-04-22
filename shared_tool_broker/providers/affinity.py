from __future__ import annotations

import base64
from typing import Any

import httpx

from ..config import BrokerSettings
from ..core import IdempotencyStore, ToolExecutionError, result_payload


class AffinityProvider:
    def __init__(self, settings: BrokerSettings, store: IdempotencyStore) -> None:
        self.settings = settings
        self.store = store

    def _api_key(self) -> str:
        key = self.settings.affinity_api_key
        if key:
            return key
        file_path = self.settings.hermes_home.parent / ".config" / "affinity" / "api_key"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8").strip()
        return "placeholder"

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, json_body: Any | None = None) -> Any:
        token = base64.b64encode(f":{self._api_key()}".encode()).decode()
        response = httpx.request(
            method,
            f"https://api.affinity.co{path}",
            headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
            params=params,
            json=json_body,
            timeout=40.0,
            trust_env=True,
        )
        if response.status_code >= 400:
            raise ToolExecutionError(f"Affinity API {response.status_code}: {response.text[:800]}", category="provider")
        return response.json() if response.text else {}

    def _search(self, resource: str, term: str) -> Any:
        return self._request("GET", f"/{resource}", params={"term": term})

    def person_search(self, *, query: str) -> dict[str, Any]:
        return result_payload(data=self._search("persons", query))

    def person_get(self, *, person_id: int) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/persons/{person_id}"))

    def person_upsert(self, *, first_name: str, last_name: str, email: str | None = None, organization_id: int | None = None, dry_run: bool = True, idempotency_key: str | None = None) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("affinity.person.upsert", idempotency_key)
            if cached is not None:
                return cached
        candidates = self._search("persons", email or f"{first_name} {last_name}")
        persons = candidates.get("persons", [])
        exact = None
        if email:
            lowered = email.lower()
            exact = next((item for item in persons if lowered in {value.lower() for value in item.get("emails", [])}), None)
        payload = {"first_name": first_name, "last_name": last_name}
        if email:
            payload["emails"] = [email]
        if organization_id:
            payload["organization_ids"] = [organization_id]
        if exact:
            result = result_payload(data={"matched": True, "confidence": "high", "person": exact})
        elif dry_run:
            result = result_payload(data={"dry_run": True, "matched": False, "candidates": persons[:5], "payload": payload})
        else:
            result = result_payload(data=self._request("POST", "/persons", json_body=payload))
        if idempotency_key:
            self.store.put("affinity.person.upsert", idempotency_key, result)
        return result

    def organization_search(self, *, query: str) -> dict[str, Any]:
        return result_payload(data=self._search("organizations", query))

    def organization_get(self, *, organization_id: int) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/organizations/{organization_id}"))

    def organization_upsert(self, *, name: str, domain: str | None = None, dry_run: bool = True, idempotency_key: str | None = None) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("affinity.organization.upsert", idempotency_key)
            if cached is not None:
                return cached
        candidates = self._search("organizations", name)
        orgs = candidates.get("organizations", [])
        exact = next((item for item in orgs if item.get("name", "").strip().lower() == name.strip().lower()), None)
        payload = {"name": name}
        if domain:
            payload["domain"] = domain
        if exact:
            result = result_payload(data={"matched": True, "confidence": "high", "organization": exact})
        elif dry_run:
            result = result_payload(data={"dry_run": True, "matched": False, "candidates": orgs[:5], "payload": payload})
        else:
            result = result_payload(data=self._request("POST", "/organizations", json_body=payload))
        if idempotency_key:
            self.store.put("affinity.organization.upsert", idempotency_key, result)
        return result

    def opportunity_search(self, *, query: str) -> dict[str, Any]:
        return result_payload(data=self._search("opportunities", query))

    def opportunity_get(self, *, opportunity_id: int) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/opportunities/{opportunity_id}"))

    def opportunity_update_stage(self, *, opportunity_id: int, stage_id: int, dry_run: bool = True, idempotency_key: str | None = None) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("affinity.opportunity.update_stage", idempotency_key)
            if cached is not None:
                return cached
        if dry_run:
            result = result_payload(data={"dry_run": True, "opportunity_id": opportunity_id, "stage_id": stage_id})
        else:
            result = result_payload(data=self._request("PUT", f"/opportunities/{opportunity_id}", json_body={"stage_id": stage_id}))
        if idempotency_key:
            self.store.put("affinity.opportunity.update_stage", idempotency_key, result)
        return result

    def note_create(
        self,
        *,
        content: str,
        person_ids: list[int] | None = None,
        organization_ids: list[int] | None = None,
        opportunity_ids: list[int] | None = None,
        dry_run: bool = True,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if idempotency_key:
            cached = self.store.get("affinity.note.create", idempotency_key)
            if cached is not None:
                return cached
        payload: dict[str, Any] = {"content": content, "type": 0}
        if person_ids:
            payload["person_ids"] = person_ids
        if organization_ids:
            payload["organization_ids"] = organization_ids
        if opportunity_ids:
            payload["opportunity_ids"] = opportunity_ids
        if dry_run:
            result = result_payload(data={"dry_run": True, "payload": payload})
        else:
            result = result_payload(data=self._request("POST", "/notes", json_body=payload))
        if idempotency_key:
            self.store.put("affinity.note.create", idempotency_key, result)
        return result
