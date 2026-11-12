"""Enumerate the search queries to run per broker, across ALL of a subject's identifiers.

People-search sites index a person under every name, phone, email, and address they
have. A subject with two names (maiden/married) and three past cities can have many
distinct listings on one broker, each found via a different search. `search_vectors`
expands the dossier into the concrete searches to run, filtered by what each broker
supports (`broker.search.by`, default ["name"]).
"""
from __future__ import annotations

import dossier as dossier_mod

# What a broker can be searched by; default if a record doesn't declare it.
DEFAULT_BY = ["name"]


def supported_by(broker: dict) -> list[str]:
    return list((broker.get("search") or {}).get("by") or DEFAULT_BY)


def search_vectors(subject_dossier: dict, broker: dict) -> list[dict]:
    """List of {by, query} searches to run for this subject on this broker."""
    by = set(supported_by(broker))
    ident = subject_dossier.get("identity", {})
    vectors: list[dict] = []

    if "name" in by:
        names = dossier_mod.all_names(subject_dossier)
        locations = dossier_mod.all_locations(subject_dossier)
        if locations:
            for name in names:
                for loc in locations:
                    vectors.append({"by": "name",
                                    "query": {"full_name": name, "city": loc.get("city"), "state": loc.get("state")}})
        else:
            for name in names:
                vectors.append({"by": "name", "query": {"full_name": name}})

    if "phone" in by:
        for phone in ident.get("phones") or []:
            vectors.append({"by": "phone", "query": {"phone": phone}})

    if "email" in by:
        for email in ident.get("emails") or []:
            vectors.append({"by": "email", "query": {"email": email}})

    if "address" in by:
        for a in dossier_mod.all_addresses(subject_dossier):
            if a.get("line1"):
                vectors.append({"by": "address",
                                "query": {k: a.get(k) for k in ("line1", "city", "state", "postal")}})

    return vectors
