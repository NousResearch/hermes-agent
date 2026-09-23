#!/usr/bin/env python3
"""Classify a Bedrock error message into the NOVA taxonomy (see references/bedrock.md §4).

  echo "$ERR" | python3 classify_bedrock_error.py          # prints JSON
  python3 classify_bedrock_error.py "An error occurred (ValidationException) ... Operation not allowed"

Order matters: specific signatures are checked before generic ones (e.g. "Operation not allowed"
is a ValidationException but means account authorization, not a schema error).
"""
import json
import re
import sys

RULES = [
    ("D", "account/model authorization", "AWS account admin / AWS Support", False,
     [r"operation not allowed", r"account is not authorized", r"don'?t have access to the model",
      r"model access", r"use case details", r"not subscribed", r"aws-marketplace"]),
    ("A", "IAM AccessDenied", "NOVA IAM (check role policy, SCP, VPC endpoint policy)", False,
     [r"accessdenied", r"not authorized to perform: bedrock:", r"explicit deny"]),
    ("C", "inference profile unavailable / required", "deployment config", False,
     [r"on-demand throughput isn.?t supported", r"inference profile", r"invalid.*profile"]),
    ("E", "region mismatch", "deployment config", False,
     [r"not supported in (this|the) region", r"region.*(mismatch|not enabled)", r"endpoint.*region"]),
    ("B", "model unavailable", "deployment config", False,
     [r"resourcenotfound", r"model identifier is invalid", r"end of life", r"legacy", r"could not resolve the foundation model"]),
    ("G", "quota/throttling", "quotas + backoff", True,
     [r"throttlingexception", r"too many requests", r"servicequotaexceeded", r"rate exceeded"]),
    ("H", "model provider restriction", "account admin", False,
     [r"provider.*(restrict|not available)", r"not available in your (country|geography)"]),
    ("I", "service-side", "AWS (retry with capped backoff, check Health dashboard)", True,
     [r"modelerrorexception", r"serviceunavailable", r"internalserver", r"model(stream)?error", r"\b5\d\d\b", r"timed? ?out"]),
    ("F", "request schema", "NOVA code", False,
     [r"validationexception", r"malformed", r"messages?\.\d+", r"max_tokens", r"required key", r"extraneous key"]),
]


def classify(msg):
    m = (msg or "").lower()
    for cat, name, owner, retryable, pats in RULES:
        for p in pats:
            if re.search(p, m):
                return {"category": cat, "name": name, "owner": owner, "retryable": retryable, "matched": p}
    return {"category": "?", "name": "unclassified", "owner": "investigate — capture full error + request id",
            "retryable": False, "matched": None}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
    print(json.dumps(classify(text), indent=2))
