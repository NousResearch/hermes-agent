#!/usr/bin/env python3
"""
Clinical Trials CLI — search and analyze clinical trials via ClinicalTrials.gov API v2.

Usage:
    python3 clinical_trials.py search "COVID-19" --limit 5
    python3 clinical_trials.py drug "Paxlovid" --status RECRUITING
    python3 clinical_trials.py detail NCT05373043
    python3 clinical_trials.py sponsor "Pfizer" --phase 3 --limit 10

No API key required. Uses ClinicalTrials.gov API v2.
"""

import json
import sys
import urllib.parse
import urllib.request

API_BASE = "https://clinicaltrials.gov/api/v2"
USER_AGENT = "Mozilla/5.0"


def api_request(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def search_trials(params: dict, limit: int = 10) -> dict:
    params["pageSize"] = min(limit, 100)
    url = f"{API_BASE}/studies?{urllib.parse.urlencode(params)}"
    data = api_request(url)
    studies = data.get("studies", [])
    total = data.get("totalCount", len(studies))
    
    results = []
    for s in studies[:limit]:
        pm = s.get("protocolSection", {})
        id_mod = pm.get("identificationModule", {})
        status_mod = pm.get("statusModule", {})
        design_mod = pm.get("designModule", {})
        sponsor_mod = pm.get("sponsorCollaboratorsModule", {})
        
        results.append({
            "nct_id": id_mod.get("nctId", ""),
            "title": id_mod.get("briefTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
            "phase": ", ".join(design_mod.get("phases", [])) if design_mod.get("phases") else "",
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
        })
    
    return {"total": total, "results": results}


def get_trial_detail(nct_id: str) -> dict:
    url = f"{API_BASE}/studies/{nct_id}"
    data = api_request(url)
    s = data  # API v2 returns study directly, no wrapper
    pm = s.get("protocolSection", {})
    
    id_mod = pm.get("identificationModule", {})
    status_mod = pm.get("statusModule", {})
    design_mod = pm.get("designModule", {})
    sponsor_mod = pm.get("sponsorCollaboratorsModule", {})
    eligibility_mod = pm.get("eligibilityModule", {})
    conditions_mod = pm.get("conditionsModule", {})
    outcome_mod = pm.get("outcomesModule", {})
    arm_mod = pm.get("armsInterventionsModule", {})
    
    detail = {
        "nct_id": id_mod.get("nctId"),
        "title": id_mod.get("briefTitle"),
        "official_title": id_mod.get("officialTitle"),
        "status": status_mod.get("overallStatus"),
        "start_date": status_mod.get("startDateStruct", {}).get("date"),
        "completion_date": status_mod.get("completionDateStruct", {}).get("date"),
        "phase": ", ".join(design_mod.get("phases", [])) if design_mod.get("phases") else "",
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name"),
        "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
        "conditions": conditions_mod.get("conditions", []),
        "eligibility": {
            "criteria": (eligibility_mod.get("eligibilityCriteria", "") or "")[:2000],
            "sex": eligibility_mod.get("sex"),
            "min_age": eligibility_mod.get("minimumAge"),
            "max_age": eligibility_mod.get("maximumAge"),
            "healthy_volunteers": eligibility_mod.get("healthyVolunteers"),
        },
        "primary_outcomes": [
            {"measure": o.get("measure"), "time_frame": o.get("timeFrame")}
            for o in (outcome_mod.get("primaryOutcomes", []) or [])
        ],
        "secondary_outcomes": [
            {"measure": o.get("measure"), "time_frame": o.get("timeFrame")}
            for o in (outcome_mod.get("secondaryOutcomes", []) or [])
        ],
        "interventions": [
            {"name": a.get("name"), "type": a.get("type")}
            for a in (arm_mod.get("interventions", []) or [])
        ],
        "locations": [],
    }
    
    locations_mod = pm.get("contactsLocationsModule", {})
    for loc in (locations_mod.get("locations", []) or []):
        detail["locations"].append({
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "country": loc.get("country"),
            "status": loc.get("status"),
        })
    
    return detail


def cmd_search(args):
    params = {"query.term": args.query, "sort": "LastUpdatePostDate"}
    if args.status:
        status_map = {"ACTIVE": "ACTIVE_NOT_RECRUITING", "RECRUITING": "RECRUITING",
                      "COMPLETED": "COMPLETED", "TERMINATED": "TERMINATED",
                      "ALL": ""}
        s = status_map.get(args.status.upper(), args.status)
        if s:
            params["filter.overallStatus"] = s
    if args.phase:
        params["filter.phase"] = args.phase
    
    result = search_trials(params, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_drug(args):
    params = {"query.term": args.name, "query.treatments": args.name}
    if args.status:
        params["filter.overallStatus"] = args.status.upper()
    result = search_trials(params, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_detail(args):
    result = get_trial_detail(args.nct_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_sponsor(args):
    params = {"query.term": args.name, "query.sponsor": args.name}
    if args.phase:
        params["filter.phase"] = args.phase
    if args.status:
        params["filter.overallStatus"] = args.status.upper()
    result = search_trials(params, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    command = sys.argv[1]

    if command == "search":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("query")
        p.add_argument("--status", choices=["ACTIVE", "RECRUITING", "COMPLETED", "TERMINATED"])
        p.add_argument("--phase", choices=["1", "2", "3", "4"])
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_search(args)
    elif command == "drug":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("name")
        p.add_argument("--status")
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_drug(args)
    elif command == "detail":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("nct_id")
        args = p.parse_args(sys.argv[2:])
        cmd_detail(args)
    elif command == "sponsor":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("name")
        p.add_argument("--phase", choices=["1", "2", "3", "4"])
        p.add_argument("--status")
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_sponsor(args)
    elif command in ("--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()