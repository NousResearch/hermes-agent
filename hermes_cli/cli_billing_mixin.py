"""Billing and subscription handlers for the interactive CLI (god-file decomposition).

This module hosts the Nous billing/subscription methods lifted out of
``cli.py``'s ``HermesCLI`` class. ``HermesCLI`` inherits
``CLIBillingMixin`` so every ``self.<handler>`` call resolves unchanged
via the MRO — behavior-neutral apart from focused billing fixes.

Import discipline mirrors ``hermes_cli.cli_commands_mixin``:
  * Neutral, non-cyclic dependencies are imported at module top level below.
  * cli.py-internal symbols (the ``_cprint``/``_b``/``_d`` helpers and
    display constants) are imported LAZILY inside each method via
    ``from cli import ...``. The mixin never imports ``cli`` at module load
    time, avoiding the cycle created when ``cli.py`` imports this mixin.
"""

from __future__ import annotations

import time


class CLIBillingMixin:
    """Mixin holding interactive-CLI billing and subscription handlers."""

    def _print_nous_credits_block(self) -> bool:
        """Print the Nous dollar balance block (two-bar view) when a Nous account
        is logged in. Returns True if it printed anything.

        Prefers the shared dollar usage model (``agent.billing_usage`` — two-bar
        plan/top-up view, dollars-only, the /usage + /subscription source of
        truth). Falls back to the legacy ``nous_credits_lines`` text only when the
        model is unavailable. Agent-independent (a portal fetch gated on "a Nous
        account is logged in"), so /usage shows the block even in the TUI
        slash-worker subprocess that resumes WITHOUT a live agent. Fail-open and
        wall-clock-bounded; honors HERMES_DEV_CREDITS_FIXTURE for offline testing.
        """
        from cli import _cprint, _b, _d

        try:
            from agent.billing_usage import build_usage_model, format_renews

            usage = build_usage_model()
        except Exception:
            usage = None
            format_renews = None  # type: ignore

        if usage is not None and usage.available and format_renews is not None:
            printed_any = False
            plan = usage.plan_name or ("Free" if usage.status == "free" else None)
            renews_display = getattr(usage, "renews_display", None) or format_renews(usage.renews_at)
            renews = f" · renews {renews_display}" if renews_display else ""
            if plan:
                print()
                _cprint(f"  {_b(f'Plan: {plan}{renews}')}")
                printed_any = True

            # All lines below go through _cprint (same renderer as the Plan line) so
            # ordering is deterministic: raw print() and _cprint() flush to different
            # buffers under patch_stdout and interleave nondeterministically (the bar
            # would race above/below the Plan line across states). Keep one path.
            for _bar_ln in self._usage_bar_lines(usage, usage.plan_name):
                _cprint(_bar_ln)
                printed_any = True
            if usage.has_topup and usage.total_spendable_usd is not None:
                _cprint(f"  Total spendable: ${usage.total_spendable_usd:,.2f}")

            if usage.status == "free":
                _cprint(f"  {_d('> Free · free models only. Run /subscription to reach paid models.')}")
                printed_any = True
            elif usage.status == "low":
                _amt = f"${usage.total_spendable_usd:,.2f}" if usage.total_spendable_usd is not None else "under $5"
                _low = f"! Low balance · {_amt} left. Run /topup or /subscription."
                _cprint(f"  {_low}")
                printed_any = True

            if printed_any:
                return True

        # Fallback: legacy text lines (only when the model is unavailable).
        from agent.account_usage import nous_credits_lines

        lines = nous_credits_lines()
        if not lines:
            return False
        print()
        for line in lines:
            print(f"  {line}")
        return True

    def _print_usage_cta(self) -> None:
        """Print the `/usage` call-to-action pointing at /subscription + /topup.

        Mirrors the TUI's ``USAGE_CTA`` (``session.ts``) so every surface ends a
        usage read with the same nudge. Only called when a Nous account is logged
        in (the balance block printed), since both commands are Nous-account only.
        """
        from cli import _cprint, _d

        _cprint(f"  {_d('Run /subscription to change plan · /topup to add to your balance')}")

    # ------------------------------------------------------------------
    # /subscription — view plan + change it in the browser (CLI surface)
    # ------------------------------------------------------------------

    def _show_subscription(self):
        """`/subscription` (alias `/upgrade`) — view the Nous plan + browser hand-off.

        The CLI mirror of the TUI ``SubscriptionOverlay``: a read of the current
        plan, this cycle's subscription credits, renewal date, and the plans you
        could switch to — then a deep-link to NAS's own ``/manage-subscription``
        page (NOT the Stripe portal; that page routes upgrade→Checkout /
        downgrade→scheduled internally). The terminal NEVER charges for a
        subscription. Fail-open: logged-out / portal hiccup degrades to a clear
        message, never a crash. Mirrors ``_show_billing``
        discipline for the interactive-vs-text split.
        """
        from cli import _cprint, _b, _d

        from agent.subscription_view import build_subscription_state, subscription_manage_url

        state = build_subscription_state()

        if not state.logged_in:
            print()
            if state.error:
                _cprint(f"  💳 {_d(f'Could not load subscription: {state.error}')}")
            else:
                _cprint(f"  💳 {_d('Not logged into Nous Portal.')}")
                print("  Run `hermes portal` to log in, then /subscription.")
            return

        # Team context: no personal plan — teams run on a shared balance.
        if state.context == "team":
            print()
            _cprint(f"  ⚕ {_b('Team subscription')}")
            print(f"  {'─' * 41}")
            if state.org_name:
                role = (state.role or "").title()
                _org_line = f"Org: {state.org_name}{f' · {role}' if role else ''}"
                _cprint(f"  {_d(_org_line)}")
            org = state.org_name or "a team org"
            print(f"  This terminal is connected to {org}. Teams run on a shared")
            print("  balance · use /topup to add funds.")
            _cprint(f"  {_d('Personal subscriptions live on your personal account.')}")
            return

        self._subscription_overview(state, subscription_manage_url(state))

    def _subscription_overview(self, state, manage_url):
        """Print the plan read block, then the browser hand-off to manage it.

        Dollars-only (no "credits") — mirrors the TUI overlay: a status line, the
        shared two-bar dollar usage view (plan + top-up) with the plan name on
        the bar, and state-matched free/low nudges. No in-terminal tier picker —
        the only action is managing the subscription on the portal.
        """
        from cli import _cprint, _b, _d

        # Shared dollar usage model (the only source with top-up dollars).
        from agent.billing_usage import format_renews
        try:
            from agent.billing_usage import build_usage_model

            usage = build_usage_model()
        except Exception:
            usage = None

        c = state.current
        is_free = not (c and c.tier_id)
        can_change = state.can_change_plan

        plan_name = (c.tier_name or c.tier_id) if c else (usage.plan_name if usage else None)
        u_status = getattr(usage, "status", None) if usage else None
        view_only = not can_change
        renews_display = getattr(usage, "renews_display", None) if usage else None
        if not renews_display and c and c.cycle_ends_at:
            renews_display = format_renews(c.cycle_ends_at)

        # Status line — dollars-only, with a "→ Plus" echo of a pending change so
        # the headline itself carries a scheduled downgrade/cancellation.
        _flip = ""
        if c and c.cancel_at_period_end:
            _flip = " → cancels"
        elif c and c.pending_downgrade_tier_name:
            _flip = f" → {c.pending_downgrade_tier_name}"
        if not plan_name:
            status = "Plan: Free · free models only"
        elif usage is not None and u_status == "low" and usage.total_spendable_usd is not None:
            _tot = f"${usage.total_spendable_usd:,.2f}"
            status = f"Plan: {plan_name}{_flip} · {_tot} left"
        else:
            _spend = getattr(usage, "total_spendable_usd", None) if usage else None
            _left = f" · ${_spend:,.2f} left" if _spend is not None else ""
            _tail = " · view only" if view_only else (f" · renews {renews_display}" if renews_display else "")
            status = f"Plan: {plan_name}{_flip}{_left}{_tail}"

        # Lead with the scheduled change (cancel > downgrade) so it can't read as
        # "nothing happened" — mirrors the TUI banner. All-`_cprint` (blanks
        # included) so the block orders deterministically even when piped.
        _trans = None
        if c and c.cancel_at_period_end:
            _when = format_renews(c.cancellation_effective_at) or "the end of the billing period"
            _trans = ((c.tier_name or "your plan"), "cancels", _when)
        elif c and c.pending_downgrade_tier_name:
            _when = format_renews(c.pending_downgrade_at) or "the end of the cycle"
            _trans = ((c.tier_name or "your plan"), c.pending_downgrade_tier_name, _when)
        _cprint("")
        if _trans:
            _from, _to, _when = _trans
            _cprint(f"  ⏳ {_b('Scheduled change')}")
            _cprint(f"  {_from} ──▶ {_to}  {_d('· ' + _when)}")
            _cprint(f"  {_d(f'You keep {_from} (and its credits) until then.')}")
            _cprint("")

        _cprint(f"  ⚕ {_b(status)}")
        print(f"  {'─' * 41}")

        # Two-bar dollar usage view — plan name labels the plan bar.
        for _bar_ln in self._usage_bar_lines(usage, plan_name):
            print(_bar_ln)
        if usage and getattr(usage, "has_topup", False) and getattr(usage, "total_spendable_usd", None) is not None:
            print(f"  Total spendable: ${usage.total_spendable_usd:,.2f}")

        # State-matched nudge (free upsell / low alert; healthy stays silent).
        if is_free:
            _cprint(f"  {_d('> Paid models need a subscription. Start one to reach them.')}")
        elif u_status == "low":
            _amt = f"${usage.total_spendable_usd:,.2f}" if usage is not None and usage.total_spendable_usd is not None else "under $5"
            _low = f"! Low balance · {_amt} left. Top up or upgrade before a mid-run cutoff."
            _cprint(f"  {_low}")

        if state.org_name:
            role = (state.role or "").title()
            _org_line = f"Org: {state.org_name}{f' · {role}' if role else ''}"
            _cprint(f"  {_d(_org_line)}")
        print(f"  {'─' * 41}")

        # ── Actions ── Members (non-admin) and non-interactive contexts fall back
        # to the portal hand-off; a paid admin/owner gets the full in-terminal
        # change flow (parity with the TUI overlay).
        if not can_change:
            print()
            _cprint(f"  {_d('Plan changes need an org admin/owner.')}")
            if manage_url:
                print(f"  Manage on portal: {manage_url}")
            return

        if not getattr(self, "_app", None):
            # Non-interactive (TUI slash-worker / piped): the modal can't run.
            print()
            if manage_url:
                print(f"  Manage your subscription: {manage_url}")
                print("  Open it in your browser, then re-run /subscription.")
            return

        if is_free:
            # Starting a NEW subscription needs a fresh card — deep-link only.
            self._subscription_open_portal(state, manage_url, verb="Start a subscription")
            return

        # Paid + admin/owner + interactive → the in-terminal change flow.
        self._subscription_change_menu(state, manage_url)

    def _subscription_open_portal(self, state, manage_url, *, verb="Manage your subscription"):
        """Open / copy the manage-subscription URL — the portal hand-off."""
        from cli import _cprint, _d

        if not manage_url:
            print()
            _cprint(f"  {_d('No manage URL available — is your portal configured?')}")
            return
        print()
        choices = [
            ("open", verb, "open the subscription page in your browser"),
            ("copy", "Copy link", "copy the manage-subscription URL to your clipboard"),
            ("cancel", "Cancel", "do nothing"),
        ]
        raw = self._prompt_text_input_modal(title=verb, detail="", choices=choices)
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if choice == "open":
            opened = False
            try:
                import webbrowser

                opened = webbrowser.open(manage_url)
            except Exception:
                opened = False
            if not opened:
                print(f"  Open this URL: {manage_url}")
            print()
            print("  Finish in your browser, then re-run /subscription.")
        elif choice == "copy":
            try:
                self._write_osc52_clipboard(manage_url)
                print(f"  📋 Copied: {manage_url}")
            except Exception:
                print(f"  Manage URL: {manage_url}")
        else:
            print("  🟡 Cancelled.")

    def _subscription_change_menu(self, state, manage_url):
        """The in-terminal change menu for a paid admin/owner (interactive)."""
        c = state.current
        has_pending = bool(c and (c.cancel_at_period_end or c.pending_downgrade_tier_name))
        keep_name = (c.tier_name if c else None) or "your plan"
        # When a change is already scheduled, undo is the most likely next intent →
        # promote it first (parity with the TUI). The Close row uses value "close"
        # (not "cancel") so typing the word "cancel" — which the alias table would
        # map to a Close row — can't be confused with "Cancel subscription".
        if has_pending:
            choices = [
                ("keep", f"Keep {keep_name} (undo the scheduled change)", "cancel the pending change"),
                ("change", "Change plan", "upgrade or downgrade in the terminal"),
            ]
        else:
            choices = [
                ("change", "Change plan", "upgrade or downgrade in the terminal"),
                ("cancel_sub", "Cancel subscription", "schedule cancellation at period end"),
            ]
        choices.append(("portal", "Manage on portal", "open the billing page in your browser"))
        choices.append(("close", "Close", "do nothing"))
        raw = self._prompt_text_input_modal(title="Manage your subscription", detail="", choices=choices)
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if choice == "change":
            self._subscription_pick_tier(state)
        elif choice == "keep":
            self._subscription_apply(state, ("resume", None))
        elif choice == "cancel_sub":
            self._subscription_confirm_cancel(state)
        elif choice == "portal":
            self._subscription_open_portal(state, manage_url)
        else:
            print("  🟡 Closed. No plan change.")

    def _subscription_pick_tier(self, state):
        """Tier picker → preview → confirm (mirrors the TUI picker screen)."""
        from agent.billing_view import format_money

        c = state.current
        tiers = tuple(state.tiers or ())
        cur_order = next((t.tier_order for t in tiers if t.is_current), 0)
        # Selectable = enabled paid tiers other than current (free/no-sub excluded;
        # dropping to free is a cancellation, on the change menu). Sorted by price.
        selectable = sorted(
            [t for t in tiers if t.is_enabled and not t.is_current and (t.tier_order or 0) > 0],
            key=lambda t: t.tier_order or 0,
        )
        if not selectable:
            print("  No other plans are available to switch to right now.")
            return
        choices = []
        for t in selectable:
            direction = "upgrade" if (t.tier_order or 0) > cur_order else "downgrade"
            choices.append((t.tier_id, f"{t.name} · {format_money(t.dollars_per_month)}/mo · {direction}", f"switch to {t.name}"))
        choices.append(("cancel", "Back", "do nothing"))
        raw = self._prompt_text_input_modal(
            title="Change plan",
            detail=f"Current: {c.tier_name if c else 'Free'}. Pick a plan to preview the effect.",
            choices=choices,
        )
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if not choice or choice == "cancel":
            print("  🟡 Cancelled. No plan change.")
            return
        self._subscription_preview_and_confirm(state, choice)

    def _subscription_preview_and_confirm(self, state, tier_id, *, allow_stepup=True):
        """Preview the change (chargeless quote), show the effect, then confirm+apply.

        ``allow_stepup=False`` (a post-grant replay) declines a second step-up on a
        repeated scope denial so the flow can't re-prompt/re-open the browser in a
        loop.
        """
        from cli import _cprint, _b, _d

        from agent.subscription_view import subscription_change_preview_from_payload
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, post_subscription_preview

        _cprint(f"  {_d('Checking the change…')}")
        try:
            payload = post_subscription_preview(subscription_type_id=tier_id)
        except BillingScopeRequired:
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=("preview", tier_id))
            else:
                print("  Remote Spending still isn't active for this terminal — the authorization didn't take. Retry, or make this change on the portal.")
            return
        except BillingError as exc:
            self._subscription_render_error(state, exc)
            return
        p = subscription_change_preview_from_payload(payload)
        effect = p.effect
        target = p.target_tier_name or "the selected plan"
        print()
        if effect == "no_op":
            _cprint(f"  {_d(f'You are already on {target} — nothing to change.')}")
            return
        if effect not in ("charge_now", "scheduled"):
            # blocked OR an unknown/unexpected effect → fail SAFE (never schedule a
            # real change on an unrecognized string, unlike a bare `else`), and
            # re-offer the portal hand-off like the TUI's blocked branch.
            from agent.subscription_view import subscription_manage_url

            _cprint(f"  🟡 {p.reason or 'This change cannot be confirmed here — manage it on the portal.'}")
            _mu = subscription_manage_url(state)
            if _mu:
                print(f"  Manage on portal: {_mu}")
            return
        if effect == "charge_now":
            _amt = f"${p.amount_due_now_cents / 100:.2f}" if p.amount_due_now_cents is not None else None
            _cprint(f"  {_b('Confirm plan change')}  {_d('· charged now')}")
            if _amt:
                _cprint(f"  Upgrade to {target}. You will be charged {_amt} now (prorated).")
            else:
                _cprint(f"  Upgrade to {target}. You will be charged the prorated amount now.")
            # Best-effort: name the exact card (billing.state), but only when the
            # resolver rung matches what a subscription charge actually uses
            # (subPin / customerDefault — Stripe's own precedence). Any failure or
            # older NAS → the generic line stands.
            _card_line = "The card on your subscription will be charged."
            try:
                from agent.billing_view import build_billing_state

                _bs = build_billing_state(timeout=6.0)
                _c = _bs.card if _bs.logged_in else None
                if _c is not None and _c.resolved_via in ("subPin", "customerDefault"):
                    _card_line = f"{_c.masked} — the card on your subscription — will be charged."
            except Exception:
                pass
            _cprint(f"  {_d(_card_line)}")
            pay_label = f"Pay {_amt} & upgrade now" if _amt else "Upgrade now (prorated charge)"
            action = ("upgrade", tier_id)
            # The money-moving row is NOT the default — a bare Enter hits "Go back",
            # so a single stray keystroke can't charge the card.
            confirm_choices = [
                ("cancel", "Go back", "do not charge"),
                ("yes", pay_label, "charge + upgrade now"),
            ]
        else:  # scheduled (whitelisted above)
            _when = p.effective_at[:10] if (p.effective_at and len(p.effective_at) >= 10) else "the end of the billing period"
            _cprint(f"  {_b('Confirm plan change')}  {_d('· scheduled · not today')}")
            _cprint(f"  Change to {target} — takes effect {_when}. No charge now; you keep your current plan until then.")
            pay_label = f"Schedule change to {target}"
            action = ("schedule", tier_id)
            confirm_choices = [
                ("yes", pay_label, "apply this change"),
                ("cancel", "Go back", "do not change"),
            ]
        if p.monthly_credits_delta:
            _cprint(f"  {_d(f'Monthly credits change: {p.monthly_credits_delta}.')}")
        raw = self._prompt_text_input_modal(title=pay_label, detail="", choices=confirm_choices)
        if self._normalize_slash_confirm_choice(raw, confirm_choices) != "yes":
            print("  🟡 Cancelled. No plan change.")
            return
        self._subscription_apply(state, action, allow_stepup=allow_stepup)

    def _subscription_confirm_cancel(self, state):
        """Confirm, then schedule a cancellation at period end."""
        from cli import _cprint, _b, _d

        from agent.billing_usage import format_renews

        c = state.current
        _end = (format_renews(c.cycle_ends_at) if (c and c.cycle_ends_at) else None) or "the end of the billing period"
        print()
        _cprint(f"  {_b('Confirm cancellation')}  {_d('· scheduled · not today')}")
        _cprint(f"  Cancel {(c.tier_name if c else 'your plan')} — it stays active until {_end}, then won't renew.")
        _cprint(f"  {_d('You keep your remaining credits for this period. You can resume before it ends.')}")
        confirm_choices = [
            ("yes", "Cancel subscription", "schedule cancellation at period end"),
            ("cancel", "Go back", "keep your plan"),
        ]
        raw = self._prompt_text_input_modal(title="Cancel subscription?", detail="", choices=confirm_choices)
        if self._normalize_slash_confirm_choice(raw, confirm_choices) != "yes":
            print("  🟡 Cancelled. Your plan is unchanged.")
            return
        self._subscription_apply(state, ("cancel", None))

    def _subscription_apply(self, state, action, idempotency_key=None, *, allow_stepup=True):
        """Run the mutation for `action`, handling the scope step-up + the result.

        `action` is one of ("upgrade", tier_id) / ("schedule", tier_id) /
        ("cancel", None) / ("resume", None). insufficient_scope routes to the
        step-up and replays; the upgrade idempotency key is reused across the replay.
        ``allow_stepup=False`` (a post-grant replay) declines a second step-up on a
        repeated scope denial so the flow can't re-prompt/re-open the browser in a loop.
        """
        from cli import _cprint, _d, _DIM, _RST

        from hermes_cli.nous_billing import (
            BillingError,
            BillingTransient,
            BillingRemoteSpendingRevoked,
            BillingScopeRequired,
            BillingSessionRevoked,
            delete_subscription_pending_change,
            post_subscription_upgrade,
            put_subscription_pending_change,
        )

        kind, arg = action
        key = None
        if kind == "upgrade":
            from agent.billing_view import new_idempotency_key

            key = idempotency_key or new_idempotency_key()
        try:
            if kind == "upgrade":
                try:
                    res = post_subscription_upgrade(subscription_type_id=arg, idempotency_key=key) or {}
                except BillingScopeRequired:
                    raise  # a scope denial rejects BEFORE charging → route to the step-up
                except (BillingTransient, BillingSessionRevoked, BillingRemoteSpendingRevoked) as exc:
                    # Deterministic PRE-charge typed rejections (429 / 401 / 403) never
                    # reached Stripe → surface the CORRECT recovery (retry_after / re-login /
                    # reconnect), NOT the "maybe charged" ambiguity copy.
                    self._subscription_render_error(state, exc)
                    return
                except BillingError as exc:
                    _status = getattr(exc, "status", None)
                    _code = getattr(exc, "error", None)
                    if _code in ("network_error", "endpoint_unavailable") or _status is None or _status >= 500:
                        # Genuinely INDETERMINATE — transport / unparseable 2xx / a 5xx the
                        # server hit mid-request: NAS may have already prorated + charged.
                        # Steer to a re-check, never a blind retry (a fresh key can't dedup →
                        # a real second charge).
                        self._subscription_render_upgrade_ambiguous(exc)
                    else:
                        # A deterministic 4xx (role_required / no_payment_method / …) → the
                        # normal error copy, not "maybe charged".
                        self._subscription_render_error(state, exc)
                    return
                status = res.get("status")
                name = res.get("targetTierName") or "your new plan"
                _url = res.get("recoveryUrl")
                if status == "already_on_tier":
                    _cprint(f"  {_DIM}✓ You are already on {name}.{_RST}")
                elif status == "upgraded":
                    _cprint(f"  {_DIM}✓ Upgraded to {name}. Your new monthly credits land in a moment.{_RST}")
                elif status == "requires_action":
                    _cprint("  🟡 This upgrade needs extra verification (3DS). Finish it on the portal.")
                    if _url:
                        _cprint(f"  Portal: {_url}")
                elif status == "payment_failed":
                    _cprint("  🔴 Your card was declined. Update your payment method on the portal and try again.")
                    if _url:
                        _cprint(f"  Portal: {_url}")
                else:
                    # Unknown / absent 2xx status → also ambiguous, not a flat failure.
                    self._subscription_render_upgrade_ambiguous(None)
                return
            if kind == "schedule":
                put_subscription_pending_change(subscription_type_id=arg)
                _cprint(f"  {_DIM}✓ Scheduled — your plan doesn't change today. You keep it until the end of the billing period, then it switches.{_RST}")
            elif kind == "cancel":
                put_subscription_pending_change(cancel=True)
                _cprint(f"  {_DIM}✓ Scheduled — your plan stays active until the end of the billing period, then it cancels. Nothing changes today.{_RST}")
            elif kind == "resume":
                delete_subscription_pending_change()
                _cprint(f"  {_DIM}✓ Undone — you stay on your current plan.{_RST}")
            _cprint(f"  {_d('Re-run /subscription anytime to review it.')}")
        except BillingScopeRequired:
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=action, idempotency_key=key)
            else:
                print("  Remote Spending still isn't active for this terminal — the authorization didn't take. Retry, or make this change on the portal.")
        except BillingError as exc:
            self._subscription_render_error(state, exc)

    def _subscription_handle_scope_required(self, state, *, retry, idempotency_key=None):
        """insufficient_scope → allow remote spending (step-up), then replay `retry`.

        Mirrors _billing_handle_scope_required: the classic CLI calls
        step_up_nous_billing_scope directly (it opens the browser + blocks), then
        replays the held preview/mutation so the user never re-runs the command.
        """
        from cli import _cprint, _d, _DIM, _RST

        print()
        print("  ! One-time setup")
        _cprint(f"  {_d('To change your plan from the terminal, allow Remote Spending once. It opens your browser to authorize, then your change picks up right here.')}")
        if not getattr(self, "_app", None):
            print("  Run `hermes portal` and allow Remote Spending, then re-run /subscription.")
            return
        confirm_choices = [
            ("yes", "Allow Remote Spending", "open your browser to authorize"),
            ("no", "Not now", "cancel"),
        ]
        raw = self._prompt_text_input_modal(
            title="Allow Remote Spending",
            detail="Opens your browser to authorize this terminal.",
            choices=confirm_choices,
        )
        if self._normalize_slash_confirm_choice(raw, confirm_choices) != "yes":
            print("  No change made. Allow Remote Spending when you're ready.")
            return
        print("  Opening your browser to allow Remote Spending…")
        try:
            from hermes_cli.auth import step_up_nous_billing_scope

            granted = step_up_nous_billing_scope(open_browser=True)
        except Exception as exc:
            print(f"  Couldn't allow Remote Spending: {exc}")
            return
        if not granted:
            print("  Couldn't allow Remote Spending — an org admin or owner has to approve it for this org.")
            return
        _cprint(f"  {_DIM}✓ Remote Spending allowed.{_RST}")
        # Bust the 30s token cache so the replay uses the freshly-scoped token. The
        # cache still holds the pre-grant unscoped token, and _request only busts it
        # on a 401 (not a 403 scope denial) — without this, the replay would 403
        # again and (before the allow_stepup guard) re-prompt in a loop.
        try:
            from hermes_cli import nous_billing as _nb

            _nb.invalidate_cached_token()
        except Exception:
            pass
        # Re-fetch fresh state, then replay the held action ONCE (allow_stepup=False
        # so a repeated scope denial can't re-enter the step-up).
        from agent.subscription_view import build_subscription_state

        try:
            fresh = build_subscription_state()
        except Exception:
            fresh = state
        rkind, rarg = retry
        if rkind == "preview":
            self._subscription_preview_and_confirm(fresh, rarg, allow_stepup=False)
        else:
            self._subscription_apply(fresh, retry, idempotency_key=idempotency_key, allow_stepup=False)

    def _subscription_render_error(self, state, exc):
        """Render a subscription BillingError (a lighter _billing_render_charge_error)."""
        from cli import _cprint

        code = getattr(exc, "error", None)
        msg = str(exc) or "Something went wrong."
        if code == "insufficient_scope":
            # Defensive: the flow routes scope to the step-up before reaching here.
            _cprint("  🟡 Remote Spending isn't allowed yet. Allow it, then retry.")
        elif code in ("subscription_mutation_rejected", "preview_rejected"):
            _cprint(f"  🟡 {msg}")
        else:
            _cprint(f"  🔴 {msg}")
        _url = getattr(exc, "portal_url", None)
        if _url:
            _cprint(f"  Portal: {_url}")

    def _subscription_render_upgrade_ambiguous(self, exc):
        """A charge-route failure (transport / timeout / 500 / unknown status) is
        AMBIGUOUS — NAS may have already prorated + charged. Steer to a re-check,
        never a flat failure that invites a blind retry (mirrors the TUI's
        upgradeResult(null) — the CLI can't persist the key across a command re-run,
        so a re-check is the safe path)."""
        from cli import _cprint, _d

        _cprint("  🟡 Couldn't confirm the upgrade — your card may or may not have been charged.")
        _cprint(f"  {_d('Re-run /subscription to check your plan before trying again.')}")
        _url = getattr(exc, "portal_url", None) if exc is not None else None
        if _url:
            _cprint(f"  Portal: {_url}")

    # ------------------------------------------------------------------
    # /billing — Phase 2b Remote Spending (CLI surface, all 5 screens)
    # ------------------------------------------------------------------

    def _show_billing(self, command: str = "/topup"):
        """`/topup` — Remote Spending for Nous (one interactive modal).

        ZERO sub-commands: any argument is ignored. Bare ``/topup`` always
        opens the Overview (Screen 1), whose numbered menu is the *only* way to
        reach the Buy / Auto-reload / Monthly-limit sub-screens. (Per the unified
        UX spec §0.4 — ``/topup buy`` etc. are gone; we don't error on a stray
        arg, we just open the menu.)

        Interactive CLI uses the prompt_toolkit modal; non-interactive contexts
        (TUI slash-worker / no live app) render text + the portal deep-link, never
        prompting (the URL is the affordance), same discipline as ``_show_subscription``.
        All money is Decimal end-to-end; the terminal never collects card details.
        """
        from cli import _cprint, _d

        from agent.billing_view import build_billing_state

        state = build_billing_state()
        if not state.logged_in:
            print()
            if state.error:
                _msg = f"Couldn't load billing: {state.error}"
                _cprint(f"  💳 {_d(_msg)}")
            else:
                _cprint(f"  💳 {_d('Not logged into Nous Portal.')}")
                print("  Run `hermes portal` to log in, then /topup.")
            return

        # Any sub-arg is intentionally ignored — always open the menu.
        self._billing_overview(state)

    def _billing_portal_hint(self, state, *, reason: str = "") -> None:
        """Print a portal deep-link line (the funnel for portal-only actions)."""
        url = getattr(state, "portal_url", None)
        if not url:
            return
        if reason:
            print(f"  {reason}")
        print(f"  Manage on portal: {url}")

    def _billing_overview(self, state):
        """Screen 1 — overview: balance in title, two-bar dollar usage, action menu.

        Dollars-only (no "credits") — mirrors the TUI /topup overlay: balance
        leads in the title, the shared plan + top-up bars render below, then the
        reordered menu (Add funds first). No scope preflight — remote spending
        is discovered reactively when a charge 403s insufficient_scope.
        """
        from cli import _cprint, _b, _d

        from agent.billing_view import format_money

        # Shared dollar usage model (plan + top-up bars), same source as /usage.
        try:
            from agent.billing_usage import build_usage_model

            usage = build_usage_model()
        except Exception:
            usage = None

        print()
        _cprint(f"  💳 {_b(f'Top up · balance {format_money(state.balance_usd)}')}")
        if state.org_name:
            role = (state.role or "").title()
            _org_line = f"Org: {state.org_name}{f' · {role}' if role else ''}"
            _cprint(f"  {_d(_org_line)}")
        print(f"  {'─' * 41}")

        # Two-bar dollar usage view (plan name on the plan bar; top-up below).
        for _bar_ln in self._usage_bar_lines(usage, getattr(usage, "plan_name", None)):
            print(_bar_ln)

        ar = state.auto_reload
        if ar is not None:
            if ar.enabled:
                print(
                    f"  Auto-reload: on — below {format_money(ar.threshold_usd)} "
                    f"→ reload to {format_money(ar.reload_to_usd)}"
                )
            else:
                print("  Auto-reload: off")
        # Card presence at a glance: which card a charge would use (with why —
        # "the card on your subscription"), or that none is saved. Only for the
        # full-menu case (admin + billing on) — others get the portal note below.
        if state.can_change_plan and state.cli_billing_enabled:
            if state.card is not None:
                print(f"  Card: {state.card.display}")
            else:
                _cprint(f"  {_d('No saved card on file — “Add funds” walks you through adding one.')}")
        print(f"  {'─' * 41}")

        # Action gating: admin + kill-switch for charge/auto-reload; everyone gets portal.
        if not state.can_change_plan:
            _cprint(f"  {_d('Billing actions require an org admin/owner.')}")
            self._billing_portal_hint(state)
            return
        if not state.cli_billing_enabled:
            _cprint(f"  {_d('Remote spending is off for this org.')}")
            self._billing_portal_hint(
                state,
                reason="A billing admin can turn it on from the portal's Hermes Agent page to add funds here.",
            )
            return

        # A missing card does NOT gate the whole overview — the org may already have
        # balance, auto-reload, or a limit to view/manage. The card only matters at
        # CHARGE time: "Add funds" -> _billing_buy_flow, which detects no card and
        # hands off to the portal there. So always show the full menu below.

        # Non-interactive (slash-worker / no live app): no modal, no sub-command
        # advertising — just the portal funnel (the URL is the affordance).
        if not getattr(self, "_app", None):
            self._billing_portal_hint(state)
            return

        # Add funds first, then settings, then the scopeless browser handoff.
        # No "Allow Remote Spending" item — that's discovered at pay time.
        # "Add funds" charges in-terminal against the org's portal-saved card
        # (server-held via POST /charge — no card ref leaves the client). A
        # missing card is NOT gated here: the buy flow reacts to the server's
        # no_payment_method 403 and hands off to the portal at charge time.
        choices = [
            ("buy", "Add funds", "add money to your balance"),
            ("auto", "Auto-reload", "configure automatic top-ups"),
            ("limit", "Monthly limit", "show the monthly spend cap (read-only)"),
            ("portal", "Manage on portal", "open the billing page in your browser"),
            ("cancel", "Cancel", "do nothing"),
        ]
        # The overview summary is already printed above; the modal only needs to
        # present the action menu — repeating the title/balance reads as a dupe.
        raw = self._prompt_text_input_modal(
            title="Top up your balance", detail="",
            choices=choices,
        )
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if choice == "buy":
            self._billing_buy_flow(state)
        elif choice == "auto":
            self._billing_auto_reload_flow(state)
        elif choice == "limit":
            self._billing_limit_screen(state)
        elif choice == "portal":
            self._billing_open_portal(state)
        else:
            print("  Cancelled.")

    def _usage_bar_lines(self, usage, plan_name) -> list:
        """The plan + top-up dollar bars as ready-to-print lines (filled = remaining).

        Returns [] when there's nothing to draw. The caller resolves ``plan_name``
        (the plan-bar label) and picks its own print fn — block ordering differs
        per surface (``_cprint`` vs ``print`` under patch_stdout). One source of
        truth for the bar format across /usage, /subscription, and /topup.
        """
        lines: list = []
        pb = getattr(usage, "plan_bar", None) if usage else None
        if pb is not None and pb.total_usd > 0:
            filled = max(0, min(10, round(pb.fill_fraction * 10)))
            bar = ("█" * filled) + ("░" * (10 - filled))
            pct_s = f" · {pb.pct_used}% used" if pb.pct_used is not None else ""
            label = (plan_name or "plan").ljust(8)[:8]
            lines.append(f"  {label}[{bar}]  ${pb.remaining_usd:,.2f} left of ${pb.total_usd:,.2f}{pct_s}")
        tb = getattr(usage, "topup_bar", None) if usage else None
        if tb is not None and tb.remaining_usd > 0:
            lines.append(f"  {'top-up'.ljust(8)}[{'█' * 10}]  ${tb.remaining_usd:,.2f} · never expires")
        return lines

    def _billing_open_portal(self, state):
        url = getattr(state, "portal_url", None)
        if not url:
            print("  No portal URL available.")
            return
        opened = False
        try:
            import webbrowser

            opened = webbrowser.open(url)
        except Exception:
            opened = False
        if not opened:
            print(f"  Open this URL: {url}")
        print("  Complete billing changes in the browser.")

    def _billing_require_admin(self, state) -> bool:
        """Guard charge/auto-reload entry points; print + return False if blocked."""
        from cli import _cprint, _d

        if not state.can_change_plan:
            print()
            _cprint(f"  💳 {_d('Billing actions require an org admin/owner.')}")
            self._billing_portal_hint(state)
            return False
        if not state.cli_billing_enabled:
            print()
            _cprint(f"  💳 {_d('Remote spending is off for this org.')}")
            self._billing_portal_hint(
                state,
                reason="A billing admin can turn it on from the portal's Hermes Agent page before adding funds.",
            )
            return False
        return True

    def _billing_add_card_flow(self, state):
        """No saved card → guide adding one on the portal, with a re-check loop.

        Cards are added on the portal (never in-terminal). "I've added it" re-fetches
        billing state so the purchase continues right here once the card is saved —
        this also recovers a transient miss (the card display is best-effort
        server-side). Returns the refreshed state (card present), or None to abandon.
        """
        from cli import _cprint, _b, _d, _DIM, _RST

        print()
        _cprint(f"  💳 {_b('Add a card first')}")
        _cprint("  No saved card on file.")
        _cprint(f"  {_d('Add a card once on the portal billing page — after that you can top up right from the terminal.')}")
        choices = [
            ("portal", "Add a card on the portal", "opens the billing page in your browser"),
            ("recheck", "I've added it — check again", "re-check for the card and continue"),
            ("cancel", "Back", "do nothing"),
        ]
        for _ in range(8):  # bounded: portal-open plus a handful of re-checks
            raw = self._prompt_text_input_modal(title="Add a card", detail="", choices=choices)
            choice = self._normalize_slash_confirm_choice(raw, choices)
            if choice == "portal":
                self._billing_open_portal(state)
                _cprint(f"  {_d('Add the card on the billing page, then pick “check again” here.')}")
                continue
            if choice == "recheck":
                from agent.billing_view import build_billing_state

                try:
                    fresh = build_billing_state()
                except Exception:
                    fresh = None
                if fresh is not None and fresh.logged_in:
                    state = fresh
                if state.card is not None:
                    _cprint(f"  {_DIM}✓ Card found: {state.card.display} — continuing.{_RST}")
                    return state
                print("  Still no card on file — finish adding it on the portal, then check again.")
                continue
            break
        print("  Cancelled. No funds added.")
        return None

    def _billing_buy_flow(self, state):
        """Screen 2 (preset select) → Screen 3 (confirm + charge + poll)."""
        from cli import _cprint, _b

        from agent.billing_view import format_money, validate_charge_amount

        if not self._billing_require_admin(state):
            return

        # No card / scope preflight here — that's the rejected anti-pattern. We let
        # the charge fly and react to whatever 403 the server returns: scope first
        # (insufficient_scope → in-flight reauth), then card (no_payment_method →
        # portal handoff via _billing_render_charge_error). Mirrors the server's gate
        # order; the user only hits the flow they actually need.

        # Screen 3 — preset selection.
        if not getattr(self, "_app", None):
            presets = ", ".join(format_money(p) for p in state.charge_presets)
            print()
            _cprint(f"  💳 {_b('Add funds')}")
            print(f"  Presets: {presets}")
            print("  Run this in the interactive CLI to complete a purchase.")
            self._billing_portal_hint(state)
            return

        # No card on file → the guided ADD-CARD path first (portal + re-check),
        # so the user isn't walked through picking an amount that will 403.
        # Returns refreshed state with a card, or None (abandoned).
        if state.card is None:
            state = self._billing_add_card_flow(state)
            if state is None or state.card is None:
                return

        preset_choices = []
        for p in state.charge_presets:
            preset_choices.append((str(p), format_money(p), "one-time credit purchase"))
        preset_choices.append(("custom", "Custom amount…", "enter your own amount"))
        preset_choices.append(("cancel", "Cancel", "do nothing"))

        card = state.card
        detail = f"Payment: {card.display}" if card else "No saved card on file"
        raw = self._prompt_text_input_modal(
            title="Add funds", detail=detail, choices=preset_choices,
        )
        choice = self._normalize_slash_confirm_choice(raw, preset_choices)
        if not choice or choice == "cancel":
            print("  Cancelled. No funds added.")
            return

        from decimal import Decimal

        if choice == "custom":
            entered = self._prompt_text_input("  Amount (USD): ")
            if entered is None:
                # None = cancelled (e.g. slash-worker can't prompt off-thread).
                print("  Cancelled. No funds added.")
                return
            v = validate_charge_amount(
                entered or "", min_usd=state.min_usd, max_usd=state.max_usd
            )
            if not v.ok:
                print(f"  🔴 {v.error}")
                return
            amount = v.amount
        else:
            try:
                amount = Decimal(choice)
            except Exception:
                print("  🔴 Invalid selection.")
                return

        self._billing_confirm_and_charge(state, amount)

    def _billing_confirm_and_charge(self, state, amount):
        """Screen 3 — confirm total + consent, charge, then poll to settlement."""
        from cli import _cprint, _b, _d

        from agent.billing_view import format_money, new_idempotency_key

        card = state.card
        print()
        _cprint(f"  💳 {_b('Confirm purchase')}")
        print(f"  {'─' * 41}")
        print(f"  Total: {format_money(amount)}")
        if card:
            print(f"  Payment: {card.display}")
            # Provenance-less payloads (older NAS) keep the generic line; when
            # the resolver says WHY this card, the Payment line carries it.
            if card.provenance is None:
                _cprint(f"  {_d('Your card saved on the portal will be charged.')}")
        print(f"  {'─' * 41}")
        _consent = (
            "By confirming, you allow Nous Research to charge your card."
        )
        _cprint(f"  {_d(_consent)}")

        confirm_choices = [
            ("pay", f"Pay {format_money(amount)} now", "submit the charge"),
            ("portal", "Manage on portal", "manage your card / billing in the browser"),
            ("cancel", "Go back", "do not charge"),
        ]
        if not getattr(self, "_app", None):
            print("  Run in the interactive CLI to confirm a purchase.")
            return
        raw = self._prompt_text_input_modal(
            title=f"Pay {format_money(amount)}?",
            detail=(card.display if card else "no saved card"),
            choices=confirm_choices,
        )
        choice = self._normalize_slash_confirm_choice(raw, confirm_choices)
        if choice == "portal":
            self._billing_open_portal(state)
            return
        if choice != "pay":
            print("  Cancelled. No funds added.")
            return

        # Submit the charge with a fresh idempotency key (reused on retry).
        from hermes_cli.nous_billing import (
            BillingError,
            BillingScopeRequired,
            post_charge,
        )

        key = new_idempotency_key()
        try:
            result = post_charge(amount_usd=amount, idempotency_key=key)
        except BillingScopeRequired:
            # In-flight reauth: allow remote spending, then resume THIS charge
            # (press-Enter beat) — no command re-run. Reuses the same idem key.
            self._billing_handle_scope_required(state, amount=amount, idempotency_key=key)
            return
        except BillingError as exc:
            self._billing_render_charge_error(state, exc)
            return

        charge_id = result.get("chargeId")
        if not charge_id:
            print("  🔴 No charge id returned; please check the portal.")
            return
        _cprint(f"  {_d('Charge submitted — confirming settlement…')}")
        self._billing_poll_charge(state, charge_id, amount)

    def _billing_poll_charge(self, state, charge_id, amount):
        """Poll loop: 2s interval, 5-min cap, cancellable. settled = ledger truth."""
        import time as _time

        from agent.billing_view import format_money
        from hermes_cli.nous_billing import (
            BillingError,
            BillingTransient,
            get_charge_status,
        )

        deadline = _time.time() + 300  # 5-minute cap
        interval = 2.0
        while _time.time() < deadline:
            try:
                status = get_charge_status(charge_id)
            except BillingTransient as exc:
                # Retry-after, NOT a failure — back off and keep polling.
                wait = exc.retry_after or 5
                _time.sleep(min(wait, 30))
                continue
            except BillingError as exc:
                print(f"  🔴 Could not check the charge: {exc}")
                return

            state_str = status.get("status")
            if state_str == "settled":
                amt = status.get("amountUsd")
                from agent.billing_view import parse_money

                shown = format_money(parse_money(amt)) if amt else format_money(amount)
                print(f"  ✓ {shown} added to your balance.")
                return
            if state_str == "failed":
                self._billing_render_charge_failed(state, status.get("reason"))
                return
            # pending → wait and poll again
            _time.sleep(interval)

        # Past the cap with no terminal state = timeout (not an error).
        print("  🟡 Still processing after 5 minutes — this is a timeout, not a "
              "failure. Check /billing or the portal shortly.")
        self._billing_portal_hint(state)

    def _billing_render_charge_failed(self, state, reason):
        """Branch the poll `failed` reasons to the right copy + portal funnel."""
        reason = (reason or "").strip()
        if reason == "authentication_required":
            print("  🔴 Your bank requires verification (3DS). Complete it on the "
                  "portal to finish this purchase.")
        elif reason == "payment_method_expired":
            print("  🔴 Your card has expired. Update it on the portal.")
        elif reason == "card_declined":
            print("  🔴 Your card was declined. Try another card on the portal.")
        else:
            print(f"  🔴 The charge didn't go through ({reason or 'processing_error'}).")
        self._billing_portal_hint(state)

    def _billing_render_charge_error(self, state, exc):
        """Render a typed BillingError at submit time (pre-poll)."""
        from hermes_cli.nous_billing import (
            BillingTransient,
            BillingRemoteSpendingRevoked,
            BillingSessionRevoked,
        )

        code = getattr(exc, "error", None)
        actor = getattr(exc, "actor", None)
        portal_url = getattr(exc, "portal_url", None) or getattr(state, "portal_url", None)
        if isinstance(exc, BillingRemoteSpendingRevoked) or code == "remote_spending_revoked":
            # CF-4: this terminal's spend was revoked. Recovery is reconnect.
            who = ("An admin stopped this terminal's spending."
                   if actor == "admin"
                   else "You stopped this terminal's spending.")
            print(f"  🔴 {who} Reconnect to restore — run `hermes portal` to re-authorize.")
        elif isinstance(exc, BillingSessionRevoked) or code == "session_revoked":
            print("  🔴 Your session was logged out. Run `hermes portal` to log in again.")
        elif code == "no_payment_method":
            print("  💳 No card on file — top up and manage billing on the portal.")
        elif code in ("cli_billing_disabled", "remote_spending_disabled") or \
                getattr(exc, "code", None) == "remote_spending_disabled":
            print("  Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page.")
        elif code == "role_required":
            print("  Adding funds needs an org admin/owner. Ask an admin, or manage on the portal.")
        elif code == "idempotency_conflict":
            print("  🔴 That charge key was already used for a different amount. Start a fresh top-up.")
        elif code == "monthly_cap_exceeded":
            remaining = (getattr(exc, "payload", {}) or {}).get("remainingUsd")
            if remaining is not None:
                print(f"  🔴 Monthly spend cap reached — ${remaining} headroom left.")
            else:
                print("  🔴 Monthly spend cap reached.")
        elif isinstance(exc, BillingTransient):
            wait = getattr(exc, "retry_after", None)
            mins = f" (try again in ~{max(1, round(wait / 60))} min)" if wait else ""
            print(f"  🟡 Too many charges right now{mins}. This isn't a payment failure.")
        elif code == "insufficient_scope":
            # Never leak the raw billing:manage scope (the post-grant replay can
            # re-raise it if the grant raced) — the concept is "Remote Spending".
            print("  🔴 Remote Spending needs approval — run /topup to allow it, then retry.")
        else:
            print(f"  🔴 {exc}")
        if portal_url:
            print(f"  Portal: {portal_url}")

    def _billing_handle_scope_required(self, state, *, amount=None, idempotency_key=None):
        """403 insufficient_scope → in-flight reauth, then resume the held charge.

        The buy path discovers remote spending isn't allowed only when the
        charge 403s — there is no preflight. We allow it in-flight ("Allow
        Remote Spending" → browser device-flow), then on return ask the user to
        press Enter to resume the held ``amount`` (reusing ``idempotency_key`` so
        the resumed charge collapses with the original). Never leaks the raw
        billing:manage scope.
        """
        from cli import _cprint, _d

        from agent.billing_view import format_money

        amount_str = format_money(amount) if amount is not None else "your top-up"
        print()
        print("  ! One-time setup")
        _cprint(f"  {_d(f'To charge from this terminal, allow Remote Spending once. It opens your browser to authorize, then {amount_str} picks up right here.')}")
        if not getattr(self, "_app", None):
            print("  Run `hermes portal` and allow Remote Spending, then retry.")
            return
        confirm_choices = [
            ("yes", "Allow Remote Spending", "open your browser to authorize"),
            ("no", "Not now", "cancel"),
        ]
        raw = self._prompt_text_input_modal(
            title="Allow Remote Spending",
            detail="Opens your browser to authorize this terminal.",
            choices=confirm_choices,
        )
        choice = self._normalize_slash_confirm_choice(raw, confirm_choices)
        if choice != "yes":
            print("  No charge made. Run /topup when you want to allow Remote Spending.")
            return
        print("  Opening your browser to allow Remote Spending…")
        try:
            from hermes_cli.auth import step_up_nous_billing_scope

            granted = step_up_nous_billing_scope(open_browser=True)
        except Exception as exc:
            print(f"  Couldn't allow Remote Spending: {exc}")
            return
        if not granted:
            print("  Couldn't allow Remote Spending — an org admin or owner has to approve it. Your card was not charged.")
            return

        # Granted. The token now carries the scope, but the ORG kill-switch
        # (cli_billing_enabled) is a separate gate — re-fetch /state so we don't
        # over-promise when a charge would still hit cli_billing_disabled.
        from agent.billing_view import build_billing_state

        fresh = build_billing_state()
        if not (fresh.logged_in and fresh.cli_billing_enabled):
            print("  Remote Spending is allowed for this terminal, but it's still off for this org. A billing admin can turn it on from the portal's Hermes Agent page, then run /topup again.")
            self._billing_portal_hint(fresh)
            return

        # Scope granted + org kill-switch on — but a charge still needs a card on
        # file. If there's none, this is a half-done state: say so and route to the
        # portal to top up / manage billing, rather than a bare "✓ enabled" that reads as done.
        if fresh.card is None:
            print("  ✓ Remote Spending allowed — but there's no card on file yet.")
            _cprint(f"  {_d('Top up and manage billing on the portal to continue.')}")
            self._billing_portal_hint(fresh)
            return

        # Nothing to resume (scope-required hit outside a charge, e.g. auto-reload
        # config) → just tell the user it's ready.
        if amount is None:
            print("  ✓ Remote Spending allowed. Run /topup to continue.")
            return

        # Press-Enter beat: the user is back from the browser; resume the held
        # purchase on an explicit confirm (reassuring, not silent).
        print("  ✓ Remote Spending allowed.")
        resume_choices = [
            ("resume", f"Resume {format_money(amount)} top-up", "finish the held purchase"),
            ("cancel", "Cancel", "do not charge"),
        ]
        raw = self._prompt_text_input_modal(
            title="Resume your top-up",
            detail=f"{format_money(amount)} is ready to finish — press Enter to resume.",
            choices=resume_choices,
        )
        if self._normalize_slash_confirm_choice(raw, resume_choices) != "resume":
            print("  Cancelled. No funds added.")
            return

        # Replay the held charge, reusing the original idempotency key so a
        # double-submit collapses to one charge.
        from hermes_cli.nous_billing import BillingError, post_charge

        from agent.billing_view import new_idempotency_key

        key = idempotency_key or new_idempotency_key()
        try:
            result = post_charge(amount_usd=amount, idempotency_key=key)
        except BillingError as exc:
            self._billing_render_charge_error(fresh, exc)
            return
        charge_id = result.get("chargeId")
        if not charge_id:
            print("  No charge id returned; please check the portal.")
            return
        _cprint(f"  {_d('Resuming your top-up — confirming settlement…')}")
        self._billing_poll_charge(fresh, charge_id, amount)

    def _billing_auto_reload_flow(self, state):
        """Screen 4 — auto-reload config: threshold + reload-to → PATCH.

        Prefills the current values from ``state.auto_reload``. Validates both
        amounts (2dp, within bounds, ``reload_to > threshold``). When auto-reload
        is already on, offers a "Turn off" path (PATCH ``enabled:false``).
        """
        from cli import _cprint, _b, _d

        from agent.billing_view import format_money, validate_charge_amount

        if not self._billing_require_admin(state):
            return

        card = state.card
        ar = state.auto_reload
        currently_on = bool(ar and ar.enabled)

        print()
        _cprint(f"  💳 {_b('Auto-reload')}")
        print(f"  {'─' * 41}")
        _cprint(f"  {_d('Automatically add funds when your balance is low.')}")
        if card:
            print(f"  Card on file: {card.masked}")
        else:
            print("  No saved card — manage billing on the portal.")
            self._billing_portal_hint(state)
            return
        if currently_on:
            print(
                f"  Currently: below {format_money(ar.threshold_usd)} → "
                f"reload to {format_money(ar.reload_to_usd)}"
            )

        if not getattr(self, "_app", None):
            print("  Run in the interactive CLI to configure auto-reload.")
            self._billing_portal_hint(state)
            return

        # When already enabled, let the user turn it off without re-entering values.
        if currently_on:
            top_choices = [
                ("edit", "Edit thresholds", "change when / how much to reload"),
                ("off", "Turn off", "disable auto-reload"),
                ("cancel", "Cancel", "do nothing"),
            ]
            raw = self._prompt_text_input_modal(
                title="Auto-reload",
                detail=(
                    f"On — below {format_money(ar.threshold_usd)} → "
                    f"reload to {format_money(ar.reload_to_usd)}"
                ),
                choices=top_choices,
            )
            top = self._normalize_slash_confirm_choice(raw, top_choices)
            if top == "off":
                self._billing_auto_reload_disable(state)
                return
            if top != "edit":
                print("  🟡 Cancelled.")
                return

        # Field 1 — threshold (prefilled when editing an existing config).
        cur_thr = format_money(ar.threshold_usd) if currently_on else None
        thr_prompt = "  When balance falls below (USD)"
        thr_prompt += f" [{cur_thr}]: " if cur_thr else ": "
        threshold_raw = self._prompt_text_input(thr_prompt)
        if threshold_raw is None:
            # None = cancelled (e.g. slash-worker can't prompt off-thread).
            print("  🟡 Cancelled.")
            return
        if not (threshold_raw or "").strip() and currently_on:
            threshold_amt = ar.threshold_usd  # keep current value on empty input
        else:
            tv = validate_charge_amount(
                threshold_raw or "", min_usd=state.min_usd, max_usd=state.max_usd
            )
            if not tv.ok or tv.amount is None:
                print(f"  🔴 {tv.error}")
                return
            threshold_amt = tv.amount

        # Field 2 — reload-to (prefilled when editing an existing config).
        cur_rel = format_money(ar.reload_to_usd) if currently_on else None
        rel_prompt = "  Reload balance to (USD)"
        rel_prompt += f" [{cur_rel}]: " if cur_rel else ": "
        reload_raw = self._prompt_text_input(rel_prompt)
        if reload_raw is None:
            print("  🟡 Cancelled.")
            return
        if not (reload_raw or "").strip() and currently_on:
            reload_amt = ar.reload_to_usd  # keep current value on empty input
        else:
            rv = validate_charge_amount(
                reload_raw or "", min_usd=state.min_usd, max_usd=state.max_usd
            )
            if not rv.ok or rv.amount is None:
                print(f"  🔴 {rv.error}")
                return
            reload_amt = rv.amount

        if reload_amt is None or threshold_amt is None or reload_amt <= threshold_amt:
            print("  🔴 Reload-to amount must be greater than the threshold.")
            return

        print()
        _ar_consent = (
            f"By confirming, you authorize Nous Research to charge {card.masked} "
            f"whenever your balance reaches {format_money(threshold_amt)}. "
            f"Turn off any time here or on the portal."
        )
        _cprint(f"  {_d(_ar_consent)}")
        confirm_choices = [
            ("agree", "Agree and turn on", "enable auto-reload"),
            ("cancel", "Cancel", "do nothing"),
        ]
        raw = self._prompt_text_input_modal(
            title="Turn on auto-reload?",
            detail=f"Below {format_money(threshold_amt)} → reload to {format_money(reload_amt)}",
            choices=confirm_choices,
        )
        choice = self._normalize_slash_confirm_choice(raw, confirm_choices)
        if choice != "agree":
            print("  🟡 Cancelled.")
            return

        from hermes_cli.nous_billing import (
            BillingError,
            BillingScopeRequired,
            patch_auto_top_up,
        )

        try:
            patch_auto_top_up(
                enabled=True, threshold=float(threshold_amt), top_up_amount=float(reload_amt)
            )
        except BillingScopeRequired:
            self._billing_handle_scope_required(state)
            return
        except BillingError as exc:
            self._billing_render_charge_error(state, exc)
            return
        print(f"  ✅ Auto-reload on: below {format_money(threshold_amt)} → "
              f"reload to {format_money(reload_amt)}.")

    def _billing_auto_reload_disable(self, state):
        """Turn off auto-reload (PATCH ``enabled:false``).

        The endpoint requires ``threshold``/``topUpAmount`` in the body even when
        disabling, so we echo back the current values (falling back to 0).
        """
        from hermes_cli.nous_billing import (
            BillingError,
            BillingScopeRequired,
            patch_auto_top_up,
        )

        ar = state.auto_reload
        thr = float(ar.threshold_usd) if ar and ar.threshold_usd is not None else 0.0
        rel = float(ar.reload_to_usd) if ar and ar.reload_to_usd is not None else 0.0
        try:
            patch_auto_top_up(enabled=False, threshold=thr, top_up_amount=rel)
        except BillingScopeRequired:
            self._billing_handle_scope_required(state)
            return
        except BillingError as exc:
            self._billing_render_charge_error(state, exc)
            return
        print("  ✅ Auto-reload turned off.")

    def _billing_limit_screen(self, state):
        """Screen 5 — monthly spend limit (read-only; cap is portal-only)."""
        from cli import _cprint, _b, _d

        from agent.billing_view import format_money

        print()
        _cprint(f"  💳 {_b('Monthly spend limit')}")
        print(f"  {'─' * 41}")
        cap = state.monthly_cap
        if cap is None or cap.limit_usd is None:
            _cprint(f"  {_d('No monthly cap visible (managed on the portal).')}")
        else:
            spent = format_money(cap.spent_this_month_usd)
            limit = format_money(cap.limit_usd)
            ceiling = " (default ceiling)" if cap.is_default_ceiling else ""
            print(f"  {spent} of {limit} used this month{ceiling}")
        _limit_note = (
            "The monthly limit is set on the portal — the terminal shows "
            "it read-only."
        )
        _cprint(f"  {_d(_limit_note)}")
        self._billing_portal_hint(state)
