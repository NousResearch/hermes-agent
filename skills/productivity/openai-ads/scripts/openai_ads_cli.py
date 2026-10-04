#!/usr/bin/env python3
"""
OpenAI Ads API (v1) Manager CLI & Automation Engine
"""
import os
import sys
import json
import argparse
import urllib.request
import urllib.error

BASE_URL = "https://api.ads.openai.com/v1"

def get_api_key():
    key = os.environ.get("OPENAI_ADS_API_KEY")
    if not key:
        print("[ERRO] OPENAI_ADS_API_KEY não definida no ambiente.", file=sys.stderr)
        sys.exit(1)
    return key

def request_api(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}" if endpoint.startswith("/") else f"{BASE_URL}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Accept": "application/json"
    }
    payload = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(data).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"[HTTP {e.code}] Erro na requisição para {endpoint}: {err_body}", file=sys.stderr)
        return {"error": True, "status": e.code, "message": err_body}
    except Exception as e:
        print(f"[ERRO] Falha de conexão: {str(e)}", file=sys.stderr)
        return {"error": True, "message": str(e)}

def cmd_account(args):
    res = request_api("/ad_account")
    print(json.dumps(res, indent=2))

def cmd_campaigns_list(args):
    res = request_api("/campaigns")
    print(json.dumps(res, indent=2))

def cmd_campaign_create(args):
    payload = {
        "name": args.name,
        "objective": args.objective or "conversions"
    }
    if args.locations:
        locs = [l.strip() for l in args.locations.split(",")]
        payload["targeting"] = {"locations": {"include": locs}}
    res = request_api("/campaigns", method="POST", data=payload)
    print(json.dumps(res, indent=2))

def cmd_adgroup_create(args):
    payload = {
        "campaign_id": args.campaign_id,
        "name": args.name,
        "bidding_type": args.bidding_type or "conversions",
        "bid_amount": int(args.bid_amount),
        "daily_budget": int(args.daily_budget) if args.daily_budget else None
    }
    if args.hints:
        payload["context_hints"] = [h.strip() for h in args.hints.split(",")]
    res = request_api("/ad_groups", method="POST", data=payload)
    print(json.dumps(res, indent=2))

def cmd_ad_create(args):
    payload = {
        "ad_group_id": args.ad_group_id,
        "name": args.name,
        "creative": {
            "type": "chat_card",
            "headline": args.headline,
            "body": args.body,
            "call_to_action": args.cta,
            "target_url": args.url
        }
    }
    if args.file_id:
        payload["creative"]["file_id"] = args.file_id
    res = request_api("/ads", method="POST", data=payload)
    print(json.dumps(res, indent=2))

def cmd_preview(args):
    res = request_api(f"/ads/{args.ad_id}/preview", method="POST")
    print(json.dumps(res, indent=2))

def cmd_insights(args):
    endpoint = f"/{args.entity}/{args.id}/insights" if args.id else f"/{args.entity}/insights"
    params = []
    if args.granularity:
        params.append(f"granularity={args.granularity}")
    if args.breakdown:
        params.append(f"breakdown={args.breakdown}")
    if params:
        endpoint += "?" + "&".join(params)
    res = request_api(endpoint)
    print(json.dumps(res, indent=2))

def main():
    parser = argparse.ArgumentParser(description="OpenAI Ads Manager CLI")
    subparsers = parser.add_subparsers(dest="command")

    p_acc = subparsers.add_parser("account", help="Informações da conta")

    p_c_list = subparsers.add_parser("campaigns-list", help="Listar campanhas")
    p_c_create = subparsers.add_parser("campaign-create", help="Criar campanha")
    p_c_create.add_argument("--name", required=True)
    p_c_create.add_argument("--objective", default="conversions")
    p_c_create.add_argument("--locations", help="Códigos de países (ex: BR,US)")

    p_ag_create = subparsers.add_parser("adgroup-create", help="Criar grupo de anúncios")
    p_ag_create.add_argument("--campaign-id", required=True)
    p_ag_create.add_argument("--name", required=True)
    p_ag_create.add_argument("--bid-amount", required=True, help="Valor em micros ($1 = 1000000)")
    p_ag_create.add_argument("--daily-budget", help="Orçamento diário em micros")
    p_ag_create.add_argument("--bidding-type", default="conversions")
    p_ag_create.add_argument("--hints", help="Context hints separados por vírgula")

    p_ad_create = subparsers.add_parser("ad-create", help="Criar anúncio")
    p_ad_create.add_argument("--ad-group-id", required=True)
    p_ad_create.add_argument("--name", required=True)
    p_ad_create.add_argument("--headline", required=True)
    p_ad_create.add_argument("--body", required=True)
    p_ad_create.add_argument("--cta", default="LEARN_MORE")
    p_ad_create.add_argument("--url", required=True)
    p_ad_create.add_argument("--file-id", help="ID do arquivo enviado via /upload")

    p_prev = subparsers.add_parser("preview", help="Preview interativo do anúncio")
    p_prev.add_argument("--ad-id", required=True)

    p_ins = subparsers.add_parser("insights", help="Relatórios e métricas")
    p_ins.add_argument("--entity", default="campaigns", choices=["campaigns", "ad_groups", "ads", "ad_account"])
    p_ins.add_argument("--id", help="ID da entidade (opcional para ad_account)")
    p_ins.add_argument("--granularity", default="day")
    p_ins.add_argument("--breakdown", help="ex: device,country,product")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "account":
        cmd_account(args)
    elif args.command == "campaigns-list":
        cmd_campaigns_list(args)
    elif args.command == "campaign-create":
        cmd_campaign_create(args)
    elif args.command == "adgroup-create":
        cmd_adgroup_create(args)
    elif args.command == "ad-create":
        cmd_ad_create(args)
    elif args.command == "preview":
        cmd_preview(args)
    elif args.command == "insights":
        cmd_insights(args)

if __name__ == "__main__":
    main()
