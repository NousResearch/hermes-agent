#!/usr/bin/env python3
"""
OpenAI Ads API Client Script
Utilitário CLI e biblioteca Python para interagir com a API OpenAI Ads v1.
"""

import os
import sys
import json
import urllib.request
import urllib.error

BASE_URL = os.getenv("OPENAI_ADS_BASE_URL", "https://api.ads.openai.com/v1")
API_KEY = os.getenv("OPENAI_ADS_API_KEY", "")

def _request(endpoint, method="GET", data=None, params=None):
    if not API_KEY:
        raise ValueError("OPENAI_ADS_API_KEY não definida no ambiente.")
    
    url = f"{BASE_URL}{endpoint}"
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f"{url}?{query_string}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read().decode("utf-8")
            return json.loads(resp_body) if resp_body else {}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        return {"error": True, "status": e.code, "message": err_msg}
    except Exception as e:
        return {"error": True, "message": str(e)}

def get_ad_account():
    return _request("/ad_account")

def list_campaigns():
    return _request("/campaigns")

def create_campaign(name, budget_micros, status="active"):
    payload = {
        "name": name,
        "status": status,
        "budget": {
            "lifetime_spend_limit_micros": budget_micros
        }
    }
    return _request("/campaigns", method="POST", data=payload)

def create_ad_group(campaign_id, name, context_hints, max_bid_micros=60000, status="active"):
    payload = {
        "campaign_id": campaign_id,
        "name": name,
        "status": status,
        "context_hints": context_hints,
        "bidding_config": {
            "billing_event_type": "impression",
            "max_bid_micros": max_bid_micros
        }
    }
    return _request("/ad_groups", method="POST", data=payload)

def upload_image(image_url):
    payload = {"image_url": image_url}
    return _request("/upload", method="POST", data=payload)

def create_ad(ad_group_id, name, title, body, target_url, file_id, status="active"):
    payload = {
        "ad_group_id": ad_group_id,
        "name": name,
        "status": status,
        "creative": {
            "type": "chat_card",
            "title": title,
            "body": body,
            "target_url": target_url,
            "file_id": file_id
        }
    }
    return _request("/ads", method="POST", data=payload)

def get_insights(ad_id, limit=7):
    return _request(f"/ads/{ad_id}/insights", params={"time_granularity": "daily", "limit": limit})

if __name__ == "__main__":
    print("OpenAI Ads Client Helper")
