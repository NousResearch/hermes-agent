"""``hermes subscription`` — AIRIES Agent Stripe billing and usage."""

from __future__ import annotations

import argparse
from typing import Callable


def build_subscription_parser(subparsers, *, cmd_subscription: Callable) -> None:
    sub = subparsers.add_parser(
        "subscription",
        aliases=["sub", "billing"],
        help="AIRIES subscription and usage limits",
        description="Manage Stripe subscriptions and view usage limits for AIRIES Agent",
    )
    sub_sub = sub.add_subparsers(dest="subscription_action")

    sub_sub.add_parser("status", help="Show plan tier and current-period usage")

    checkout = sub_sub.add_parser("checkout", help="Start Stripe Checkout for a paid tier")
    checkout.add_argument(
        "--tier",
        choices=["pro", "team"],
        default="pro",
        help="Subscription tier (default: pro)",
    )
    checkout.add_argument(
        "--success-url",
        default="http://127.0.0.1:8765/subscription/success",
        help="Redirect URL after successful payment",
    )
    checkout.add_argument(
        "--cancel-url",
        default="http://127.0.0.1:8765/subscription/cancel",
        help="Redirect URL if checkout is canceled",
    )

    portal = sub_sub.add_parser("portal", help="Open Stripe billing portal")
    portal.add_argument(
        "--return-url",
        default="http://127.0.0.1:8765/",
        help="Return URL after managing billing",
    )

    sync = sub_sub.add_parser("sync", help="Refresh subscription status from Stripe")
    activate = sub_sub.add_parser("activate", help="Apply a completed Checkout session ID")
    activate.add_argument("session_id", help="Stripe Checkout session ID (cs_...)")

    sub.set_defaults(func=cmd_subscription)
