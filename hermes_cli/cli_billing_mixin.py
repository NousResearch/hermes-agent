"""Billing and subscription handlers for the interactive CLI (mixed into ``HermesCLI``).
cli.py symbols are imported LAZILY inside methods — never at module load (import cycle).

User-facing copy is looked up through ``agent.i18n.t`` at call time (never at import — the
active language is not known when this module loads), under ``cli.billing.*`` (shared /topup
copy) and ``cli.subscription.*`` (/subscription copy). Modal choice VALUES (index 0) stay
English identifiers; only labels/descriptions are translated."""

from __future__ import annotations

from agent.i18n import t

_RULE = "─" * 41

# Poll `failed` reasons → copy key (default: generic line carrying the raw reason).
_CHARGE_FAILED_KEYS = {
    "authentication_required": "cli.billing.charge_failed_authentication_required",
    "payment_method_expired": "cli.billing.charge_failed_payment_method_expired",
    "card_declined": "cli.billing.charge_failed_card_declined"}

# Submit-time BillingError codes with a fixed copy key (no payload/type inspection).
_CHARGE_ERROR_KEYS = {
    "no_payment_method": "cli.billing.charge_error_no_payment_method",
    "cli_billing_disabled": "cli.billing.charge_error_remote_spending_disabled",
    "remote_spending_disabled": "cli.billing.charge_error_remote_spending_disabled",
    "role_required": "cli.billing.charge_error_role_required",
    "idempotency_conflict": "cli.billing.charge_error_idempotency_conflict"}

# Upgrade 2xx `status` → (copy key, echo recoveryUrl as "Portal:"). Missing status → ambiguous.
_UPGRADE_STATUS_KEYS = {
    "requires_action": ("cli.subscription.upgrade_requires_action", True),
    "payment_failed": ("cli.subscription.upgrade_payment_failed", True)}

# Upgrade 2xx terminal statuses → dim ✓ copy key ({name} = target tier).
_UPGRADE_OK_KEYS = {
    "already_on_tier": "cli.subscription.upgrade_already_on_tier",
    "upgraded": "cli.subscription.upgrade_upgraded"}
# Pending-change mutations → dim ✓ copy key.
_PENDING_OK_KEYS = {
    "schedule": "cli.subscription.pending_scheduled",
    "cancel": "cli.subscription.pending_cancel_scheduled",
    "resume": "cli.subscription.pending_resumed"}


# ── Static modal menus (value, label, description) — built at call time so labels follow the
#    active language; order is the rendered order. ──

def _cancel_row(value: str = "cancel", label_key: str = "cli.billing.choice_cancel",
                desc_key: str = "cli.billing.desc_do_nothing") -> tuple[str, str, str]:
    return (value, t(label_key), t(desc_key))


def _portal_row(value: str = "portal") -> tuple[str, str, str]:
    return (value, t("cli.billing.choice_manage_on_portal"), t("cli.billing.desc_open_billing_page"))


def _allow_remote_spending_choices() -> list[tuple[str, str, str]]:
    return [
        ("yes", t("cli.billing.choice_allow_remote_spending"), t("cli.billing.desc_allow_remote_spending")),
        ("no", t("cli.billing.choice_not_now"), t("cli.billing.desc_not_now"))]


def _topup_menu_choices() -> list[tuple[str, str, str]]:
    return [
        ("buy", t("cli.billing.choice_add_funds"), t("cli.billing.desc_add_funds")),
        ("auto", t("cli.billing.choice_auto_reload"), t("cli.billing.desc_auto_reload")),
        ("limit", t("cli.billing.choice_monthly_limit"), t("cli.billing.desc_monthly_limit")),
        _portal_row(),
        _cancel_row()]


def _add_card_choices() -> list[tuple[str, str, str]]:
    return [
        ("portal", t("cli.billing.choice_add_card_on_portal"), t("cli.billing.desc_add_card_on_portal")),
        ("recheck", t("cli.billing.choice_recheck_card"), t("cli.billing.desc_recheck_card")),
        _cancel_row(label_key="cli.billing.choice_back")]


def _auto_reload_top_choices() -> list[tuple[str, str, str]]:
    return [
        ("edit", t("cli.billing.choice_edit_thresholds"), t("cli.billing.desc_edit_thresholds")),
        ("off", t("cli.billing.choice_turn_off"), t("cli.billing.desc_turn_off")),
        _cancel_row()]


def _auto_reload_agree_choices() -> list[tuple[str, str, str]]:
    return [
        ("agree", t("cli.billing.choice_agree_turn_on"), t("cli.billing.desc_agree_turn_on")),
        _cancel_row()]


def _change_plan_row() -> tuple[str, str, str]:
    return ("change", t("cli.subscription.choice_change_plan"), t("cli.subscription.desc_change_plan"))


def _change_menu_tail() -> list[tuple[str, str, str]]:
    return [_portal_row(), _cancel_row("close", "cli.billing.choice_close")]


def _cancel_sub_choices() -> list[tuple[str, str, str]]:
    return [
        ("yes", t("cli.subscription.choice_cancel_subscription"), t("cli.subscription.desc_cancel_subscription")),
        ("cancel", t("cli.billing.choice_go_back"), t("cli.subscription.desc_keep_your_plan"))]


class CLIBillingMixin:
    """Mixin holding interactive-CLI billing and subscription handlers."""

    # ── Shared helpers ──

    def _modal_choice(self, title, detail, choices):
        """Run the choice modal and return the normalized choice value."""
        raw = self._prompt_text_input_modal(title=title, detail=detail, choices=choices)
        return self._normalize_slash_confirm_choice(raw, choices)

    def _block_header(self, icon, title, *, rule=True) -> None:
        """Blank line, ``<icon> <bold title>`` via _cprint, then (optionally) the rule via print."""
        from cli import _cprint, _b
        print()
        _cprint(f"  {icon} {_b(title)}")
        if rule:
            print(f"  {_RULE}")

    def _dim(self, msg, *, icon="", lead=False) -> None:
        """Dim ``  <icon><msg>`` line via _cprint; ``lead`` prints a blank line first."""
        from cli import _cprint, _d
        if lead:
            print()
        _cprint(f"  {icon}{_d(msg)}")

    def _ok(self, msg) -> None:
        """Dim ``✓ <msg>`` success line."""
        from cli import _cprint, _DIM, _RST
        _cprint(f"  {_DIM}✓ {msg}{_RST}")

    def _print_logged_out(self, state, load_failed, cmd) -> None:
        """Logged-out / fetch-failed block shared by /subscription and /topup."""
        if state.error:
            self._dim(t("cli.billing.load_failed", label=load_failed, error=state.error), icon="💳 ", lead=True)
        else:
            self._dim(t("cli.billing.not_logged_in"), icon="💳 ", lead=True)
            print(f"  {t('cli.billing.run_portal_then', cmd=cmd)}")

    def _print_org_line(self, state) -> None:
        """Dim ``Org: <name> · <Role>`` line (skipped when there is no org)."""
        if state.org_name:
            role = (state.role or "").title()
            if role:
                self._dim(t("cli.billing.org_line_role", org=state.org_name, role=role))
            else:
                self._dim(t("cli.billing.org_line", org=state.org_name))

    def _try_usage_model(self):
        """Shared dollar usage model (the only source with top-up dollars); None on any failure."""
        try:
            from agent.billing_usage import build_usage_model
            return build_usage_model()
        except Exception:
            return None

    def _usage_bar_lines(self, usage, plan_name) -> list:
        """Plan + top-up bars as lines (filled = remaining). Caller picks the print fn: ordering differs per surface."""
        lines: list = []
        pb = usage.plan_bar if usage else None
        if pb is not None and pb.total_usd > 0:
            filled = max(0, min(10, round(pb.fill_fraction * 10)))
            bar = ("█" * filled) + ("░" * (10 - filled))
            pct_s = t("cli.billing.bar_pct_used", pct=pb.pct_used) if pb.pct_used is not None else ""
            label = (plan_name or t("cli.billing.bar_plan_label")).ljust(8)[:8]
            lines.append("  " + t("cli.billing.bar_plan", label=label, bar=bar, remaining=f"{pb.remaining_usd:,.2f}",
                                  total=f"{pb.total_usd:,.2f}", pct=pct_s))
        tb = usage.topup_bar if usage else None
        if tb is not None and tb.remaining_usd > 0:
            lines.append("  " + t("cli.billing.bar_topup", label=t("cli.billing.bar_topup_label").ljust(8)[:8],
                                  bar="█" * 10, remaining=f"{tb.remaining_usd:,.2f}"))
        return lines

    def _print_total_spendable(self, usage, print_fn) -> None:
        if usage and usage.has_topup and usage.total_spendable_usd is not None:
            print_fn(f"  {t('cli.billing.total_spendable', amount=f'{usage.total_spendable_usd:,.2f}')}")

    def _step_up_remote_spending(self, *, explain, noninteractive_msg, declined_msg, not_granted_msg) -> bool:
        """"! One-time setup" step-up (explain → confirm → device-flow). True only when granted; refusals print."""
        print()
        print(f"  {t('cli.billing.one_time_setup')}")
        self._dim(explain)
        if not self._app:
            print(noninteractive_msg)
            return False
        choice = self._modal_choice(
            t("cli.billing.allow_remote_spending_title"), t("cli.billing.allow_remote_spending_detail"),
            _allow_remote_spending_choices())
        if choice != "yes":
            print(declined_msg)
            return False
        print(f"  {t('cli.billing.opening_browser_remote_spending')}")
        try:
            from hermes_cli.auth import step_up_nous_billing_scope
            granted = step_up_nous_billing_scope(open_browser=True)
        except Exception as exc:
            print(f"  {t('cli.billing.couldnt_allow_remote_spending', error=exc)}")
            return False
        if not granted:
            print(not_granted_msg)
        return bool(granted)

    def _print_portal_line(self, exc) -> None:
        """``Portal: <url>`` via _cprint when the error carries a portal deep-link."""
        from cli import _cprint
        if exc is not None and exc.portal_url:
            _cprint(f"  {t('cli.billing.portal_line', url=exc.portal_url)}")

    def _open_url_in_browser(self, url: str) -> bool:
        """The one portal opener. Refuses TTY-hijacking text browsers (w3m/lynx over SSH) via the auth guard."""
        if not url:
            return False
        try:
            from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session
            if _is_remote_session() or not _can_open_graphical_browser():
                return False
        except Exception:
            pass  # guard unavailable → plain best-effort open
        try:
            import webbrowser
            return bool(webbrowser.open(url))
        except Exception:
            return False

    def _open_or_print_url(self, url) -> None:
        """Open ``url`` in the browser, or print it when no graphical browser can be used."""
        if not self._open_url_in_browser(url):
            print(f"  {t('cli.billing.open_this_url', url=url)}")

    # ── /usage — Nous balance block ──

    def _print_nous_credits_block(self) -> bool:
        """Nous dollar balance block (two bars); True if anything printed. Shared dollar model first, then
        legacy ``nous_credits_lines``. Agent-independent (TUI slash-worker has no live agent). Fail-open."""
        from cli import _cprint, _b, _d
        usage = self._try_usage_model()
        if usage is not None and usage.available:
            from agent.billing_usage import format_renews
            plan = usage.plan_name or (t("cli.billing.plan_free") if usage.status == "free" else None)
            renews_display = usage.renews_display or format_renews(usage.renews_at)
            renews = t("cli.billing.renews_suffix", date=renews_display) if renews_display else ""
            head = [f"  {_b(t('cli.billing.plan_line', plan=plan, renews=renews))}"] if plan else []
            head += self._usage_bar_lines(usage, usage.plan_name)
            tail = []
            if usage.status == "free":
                tail.append(f"  {_d(t('cli.billing.free_models_only_hint'))}")
            elif usage.status == "low":
                _amt = f"${usage.total_spendable_usd:,.2f}" if usage.total_spendable_usd is not None else t("cli.billing.under_five")
                tail.append(f"  {t('cli.billing.low_balance_usage', amount=_amt)}")
            # All via _cprint like the Plan line: print()/_cprint() flush to different buffers under
            # patch_stdout. "Total spendable" alone does not count as printed (legacy lines still follow).
            if plan:
                print()
            for ln in head:
                _cprint(ln)
            self._print_total_spendable(usage, _cprint)
            for ln in tail:
                _cprint(ln)
            if head or tail:
                return True
        from agent.account_usage import nous_credits_lines
        lines = nous_credits_lines()
        if not lines:
            return False
        print()
        for line in lines:
            print(f"  {line}")
        return True

    def _print_usage_cta(self) -> None:
        """The `/usage` call-to-action; mirrors the TUI's ``USAGE_CTA``. Nous-account only."""
        self._dim(t("cli.billing.usage_cta"))

    # ── /subscription — view plan + change it (CLI surface) ──

    def _show_subscription(self):
        """`/subscription` (alias `/upgrade`). Deep-links NAS's ``/manage-subscription`` (NOT Stripe); never charges."""
        from agent.subscription_view import build_subscription_state, subscription_manage_url
        state = build_subscription_state()
        if not state.logged_in:
            self._print_logged_out(state, t("cli.subscription.load_failed_label"), "/subscription")
            return
        if state.context == "team":  # no personal plan — teams run on a shared balance
            self._block_header("☤", t("cli.subscription.team_header"))
            self._print_org_line(state)
            print(f"  {t('cli.subscription.team_connected', org=state.org_name or t('cli.subscription.a_team_org'))}")
            self._dim(t("cli.subscription.personal_note"))
            return
        self._subscription_overview(state, subscription_manage_url(state))

    def _subscription_overview(self, state, manage_url):
        """Plan read block, then the action: portal hand-off (member / non-interactive), catalog (Free), change menu."""
        from cli import _cprint, _b, _d
        from agent.billing_usage import format_renews
        usage = self._try_usage_model()
        c = state.current
        is_free = not (c and c.tier_id)
        can_change = state.can_change_plan
        plan_name = (c.tier_name or c.tier_id) if c else (usage.plan_name if usage else None)
        u_status = usage.status if usage else None
        _spend = usage.total_spendable_usd if usage else None
        renews_display = usage.renews_display if usage else None
        if not renews_display and c and c.cycle_ends_at:
            renews_display = format_renews(c.cycle_ends_at)
        # Headline flags the pending change ("→ Plus" / "→ cancels"); the banner (cancel > downgrade)
        # leads so it can't read as "nothing happened".
        _flip, _trans = "", None
        _your_plan = t("cli.subscription.your_plan")
        if c and c.cancel_at_period_end:
            _flip = t("cli.subscription.status_flip_cancels")
            _trans = (c.tier_name or _your_plan, t("cli.subscription.cancels"),
                      format_renews(c.cancellation_effective_at) or t("cli.subscription.end_of_billing_period"))
        elif c and c.pending_downgrade_tier_name:
            _flip = t("cli.subscription.status_flip_to", tier=c.pending_downgrade_tier_name)
            _trans = (c.tier_name or _your_plan, c.pending_downgrade_tier_name,
                      format_renews(c.pending_downgrade_at) or t("cli.subscription.end_of_cycle"))
        _left = t("cli.subscription.status_left", amount=f"{_spend:,.2f}") if _spend is not None else ""
        if u_status == "low" and _spend is not None:
            _tail = ""
        elif not can_change:
            _tail = t("cli.subscription.status_view_only")
        else:
            _tail = t("cli.subscription.status_renews", date=renews_display) if renews_display else ""
        if plan_name:
            status = t("cli.subscription.status_line", plan=plan_name, flip=_flip, left=_left, tail=_tail)
        else:
            status = t("cli.subscription.status_free")
        # All-_cprint (blanks included) so the block orders deterministically even when piped.
        _cprint("")
        if _trans:
            _from, _to, _when = _trans
            _cprint(f"  ⏳ {_b(t('cli.subscription.scheduled_change'))}")
            _cprint("  " + t("cli.subscription.scheduled_change_line", from_plan=_from, to_plan=_to,
                             when=_d(t("cli.subscription.scheduled_when", when=_when))))
            self._dim(t("cli.subscription.keep_until_then", plan=_from))
            _cprint("")
        _cprint(f"  ☤ {_b(status)}")
        print(f"  {_RULE}")
        for _bar_ln in self._usage_bar_lines(usage, plan_name):
            print(_bar_ln)
        self._print_total_spendable(usage, print)
        if is_free:
            self._dim(t("cli.subscription.paid_models_need_subscription"))
        elif u_status == "low":
            _amt = f"${_spend:,.2f}" if _spend is not None else t("cli.billing.under_five")
            _cprint(f"  {t('cli.subscription.low_balance', amount=_amt)}")
        self._print_org_line(state)
        print(f"  {_RULE}")
        if not can_change:
            self._dim(t("cli.subscription.plan_changes_need_admin"), lead=True)
            if manage_url:
                print(f"  {t('cli.billing.manage_on_portal_url', url=manage_url)}")
        elif not self._app:  # non-interactive (TUI slash-worker / piped): the modal can't run
            print()
            if manage_url:
                print(f"  {t('cli.subscription.manage_url_line', url=manage_url)}")
                print(f"  {t('cli.subscription.open_then_rerun')}")
        elif is_free:  # a NEW subscription needs a fresh card → catalog + portal deep-link only
            self._subscription_free_catalog(state, manage_url)
        else:
            self._subscription_change_menu(state, manage_url)

    def _subscription_free_catalog(self, state, manage_url):
        """Free + admin + interactive: catalog → pick → portal deep-link ``plan=<tier_id>`` (a new sub needs a card)."""
        from agent.subscription_view import format_tier_row, selectable_tiers, subscription_manage_url
        tiers = selectable_tiers(state)
        if not tiers:
            self._subscription_open_portal(state, manage_url, verb=t("cli.subscription.start_subscription"))
            return
        self._block_header("☤", t("cli.subscription.choose_plan"))
        for i, tier in enumerate(tiers, 1):
            print(f"  {i}. {format_tier_row(tier)}")
        self._dim(t("cli.subscription.start_opens_portal"))
        choices = [(tier.tier_id, format_tier_row(tier), t("cli.subscription.desc_start_on_portal", name=tier.name))
                   for tier in tiers]
        choices.append(_cancel_row())
        raw = self._prompt_text_input_modal(
            title=t("cli.subscription.start_subscription"), detail=t("cli.subscription.pick_plan_detail"), choices=choices)
        # Rows are numbered → accept a bare number (the normalizer only knows confirm-dialog digits).
        _digit = (raw or "").strip()
        _by_row = tiers[int(_digit) - 1] if _digit.isdigit() and 1 <= int(_digit) <= len(tiers) else None
        choice = _by_row.tier_id if _by_row else self._normalize_slash_confirm_choice(raw, choices)
        if not choice or choice == "cancel":
            print(f"  {t('cli.subscription.cancelled_no_plan_started')}")
            return
        tier_url = subscription_manage_url(state, tier_id=choice) or manage_url
        if not tier_url:
            self._dim(t("cli.billing.no_manage_url"))
            return
        picked = next((tier for tier in tiers if tier.tier_id == choice), None)
        label = picked.name if picked else t("cli.subscription.your_plan")
        if self._open_url_in_browser(tier_url):
            print(f"  {t('cli.subscription.opening_portal_to_start', plan=label)}")
        else:
            print(f"  {t('cli.subscription.open_url_to_start', plan=label, url=tier_url)}")
        print(f"  {t('cli.subscription.finish_in_browser')}")

    def _subscription_open_portal(self, state, manage_url, *, verb=None):
        """Open / copy the manage-subscription URL — the portal hand-off."""
        verb = verb or t("cli.subscription.manage_your_subscription")
        print()
        if not manage_url:
            self._dim(t("cli.billing.no_manage_url"))
            return
        choices = [
            ("open", verb, t("cli.subscription.desc_open_subscription_page")),
            ("copy", t("cli.subscription.choice_copy_link"), t("cli.subscription.desc_copy_link")),
            _cancel_row()]
        choice = self._modal_choice(verb, "", choices)
        if choice == "open":
            self._open_or_print_url(manage_url)
            print()
            print(f"  {t('cli.subscription.finish_in_browser')}")
        elif choice == "copy":
            try:
                self._write_osc52_clipboard(manage_url)
                print(f"  {t('cli.subscription.copied', url=manage_url)}")
            except Exception:
                print(f"  {t('cli.subscription.manage_url', url=manage_url)}")
        else:
            print(f"  {t('cli.billing.cancelled_yellow')}")

    def _subscription_change_menu(self, state, manage_url):
        """The in-terminal change menu for a paid admin/owner (interactive)."""
        c = state.current
        # A scheduled change makes undo the likeliest intent → promote it first. The Close row is
        # "close" (not "cancel") so typing "cancel" can't be confused with "Cancel subscription".
        if c and (c.cancel_at_period_end or c.pending_downgrade_tier_name):
            keep_name = c.tier_name or t("cli.subscription.your_plan")
            head = [("keep", t("cli.subscription.choice_keep_plan", plan=keep_name), t("cli.subscription.desc_keep_plan")),
                    _change_plan_row()]
        else:
            head = [_change_plan_row(),
                    ("cancel_sub", t("cli.subscription.choice_cancel_subscription"), t("cli.subscription.desc_cancel_subscription"))]
        choice = self._modal_choice(t("cli.subscription.manage_your_subscription"), "", head + _change_menu_tail())
        action = {
            "change": lambda: self._subscription_pick_tier(state),
            "keep": lambda: self._subscription_apply(state, ("resume", None)),
            "cancel_sub": lambda: self._subscription_confirm_cancel(state),
            "portal": lambda: self._subscription_open_portal(state, manage_url)}.get(choice)
        if action:
            action()
        else:
            print(f"  {t('cli.subscription.closed_no_change')}")

    def _subscription_pick_tier(self, state):
        """Tier picker → preview → confirm. Paid tiers other than current (dropping to free = cancellation)."""
        from agent.subscription_view import format_tier_row, is_upgrade, selectable_tiers
        c = state.current
        selectable = selectable_tiers(state)
        if not selectable:
            print(f"  {t('cli.subscription.no_other_plans')}")
            return
        choices = []
        for tier in selectable:
            direction = t("cli.subscription.direction_upgrade") if is_upgrade(state, tier.tier_id) else t("cli.subscription.direction_downgrade")
            choices.append((tier.tier_id, t("cli.subscription.tier_row", row=format_tier_row(tier), direction=direction),
                            t("cli.subscription.desc_switch_to", name=tier.name)))
        choices.append(_cancel_row(label_key="cli.billing.choice_back"))
        choice = self._modal_choice(
            t("cli.subscription.change_plan_title"),
            t("cli.subscription.current_pick_detail", plan=c.tier_name if c else t("cli.billing.plan_free")), choices)
        if not choice or choice == "cancel":
            print(f"  {t('cli.subscription.cancelled_no_plan_change')}")
            return
        self._subscription_preview_and_confirm(state, choice)

    def _subscription_preview_and_confirm(self, state, tier_id, *, allow_stepup=True):
        """Preview → effect → confirm+apply. ``allow_stepup=False`` (post-grant replay) never re-prompts a step-up."""
        from cli import _cprint, _b, _d
        from agent.subscription_view import is_upgrade, subscription_change_preview_from_payload, subscription_manage_url
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, post_subscription_preview
        self._dim(t("cli.subscription.checking_change"))
        try:
            payload = post_subscription_preview(subscription_type_id=tier_id)
        except BillingScopeRequired:
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=("preview", tier_id))
            else:
                print(f"  {t('cli.billing.stepup_stale')}")
            return
        except BillingError as exc:
            self._subscription_render_error(state, exc)
            return
        p = subscription_change_preview_from_payload(payload)
        effect = p.effect
        target = p.target_tier_name or t("cli.subscription.the_selected_plan")
        print()
        if effect == "no_op":
            self._dim(t("cli.subscription.already_on_nothing_to_change", plan=target))
            return
        if effect not in ("charge_now", "scheduled"):
            # blocked OR unknown effect → fail SAFE (never schedule on an unrecognized string) and
            # re-offer the portal. plan= rides along only for an UPGRADE hand-off (downgrades stay native).
            _cprint(f"  🟡 {p.reason or t('cli.subscription.cannot_confirm_here')}")
            _plan = tier_id if is_upgrade(state, tier_id) else None
            _mu = subscription_manage_url(state, tier_id=_plan)
            if _mu:
                print(f"  {t('cli.billing.manage_on_portal_url', url=_mu)}")
            return
        _tag = t("cli.subscription.tag_charged_now") if effect == "charge_now" else t("cli.subscription.tag_scheduled_not_today")
        _cprint(f"  {_b(t('cli.subscription.confirm_plan_change'))}  {_d(_tag)}")
        if effect == "charge_now":
            _amt = f"${p.amount_due_now_cents / 100:.2f}" if p.amount_due_now_cents is not None else None
            _charged = t("cli.subscription.charged_amount_now", amount=_amt) if _amt else t("cli.subscription.charged_prorated_now")
            _cprint(f"  {t('cli.subscription.upgrade_will_be_charged', plan=target, charged=_charged)}")
            # Best-effort: name the exact card, but only when the resolver rung matches what a
            # subscription charge actually uses (subPin / customerDefault — Stripe's precedence).
            _card_line = t("cli.subscription.card_on_subscription_charged")
            try:
                from agent.billing_view import build_billing_state
                _bs = build_billing_state(timeout=6.0)
                _c = _bs.card if _bs.logged_in else None
                if _c is not None and _c.resolved_via in ("subPin", "customerDefault"):
                    _card_line = t("cli.subscription.named_card_on_subscription_charged", card=_c.masked)
            except Exception:
                pass
            self._dim(_card_line)
            pay_label = (t("cli.subscription.choice_pay_and_upgrade", amount=_amt) if _amt
                         else t("cli.subscription.choice_upgrade_now_prorated"))
            action = ("upgrade", tier_id)
            # The money-moving row is NOT the default — a bare Enter hits "Go back", so a stray keystroke can't charge.
            confirm_choices = [
                _cancel_row(label_key="cli.billing.choice_go_back", desc_key="cli.billing.desc_do_not_charge"),
                ("yes", pay_label, t("cli.subscription.desc_charge_and_upgrade"))]
        else:  # scheduled (whitelisted above)
            _when = (p.effective_at[:10] if (p.effective_at and len(p.effective_at) >= 10)
                     else t("cli.subscription.end_of_billing_period"))
            _cprint(f"  {t('cli.subscription.change_takes_effect', plan=target, when=_when)}")
            pay_label = t("cli.subscription.choice_schedule_change", plan=target)
            action = ("schedule", tier_id)
            confirm_choices = [
                ("yes", pay_label, t("cli.subscription.desc_apply_change")),
                _cancel_row(label_key="cli.billing.choice_go_back", desc_key="cli.subscription.desc_do_not_change")]
        if p.monthly_credits_delta:
            self._dim(t("cli.subscription.monthly_credits_change", delta=p.monthly_credits_delta))
        if self._modal_choice(pay_label, "", confirm_choices) != "yes":
            print(f"  {t('cli.subscription.cancelled_no_plan_change')}")
            return
        self._subscription_apply(state, action, allow_stepup=allow_stepup)

    def _subscription_confirm_cancel(self, state):
        """Confirm, then schedule a cancellation at period end."""
        from cli import _cprint, _b, _d
        from agent.billing_usage import format_renews
        c = state.current
        _end = ((format_renews(c.cycle_ends_at) if (c and c.cycle_ends_at) else None)
                or t("cli.subscription.end_of_billing_period"))
        print()
        _cprint(f"  {_b(t('cli.subscription.confirm_cancellation'))}  {_d(t('cli.subscription.tag_scheduled_not_today'))}")
        _cprint("  " + t("cli.subscription.cancel_stays_active_until",
                         plan=(c.tier_name if c else t("cli.subscription.your_plan")), end=_end))
        self._dim(t("cli.subscription.keep_remaining_credits"))
        if self._modal_choice(t("cli.subscription.cancel_subscription_question"), "", _cancel_sub_choices()) != "yes":
            print(f"  {t('cli.subscription.cancelled_plan_unchanged')}")
            return
        self._subscription_apply(state, ("cancel", None))

    def _subscription_apply(self, state, action, idempotency_key=None, *, allow_stepup=True):
        """Run ("upgrade"|"schedule", tier_id) / ("cancel"|"resume", None); scope denial → step-up + ONE replay, same key."""
        from cli import _cprint
        from hermes_cli.nous_billing import (
            BillingError, BillingTransient, BillingRemoteSpendingRevoked, BillingScopeRequired, BillingSessionRevoked,
            delete_subscription_pending_change, post_subscription_upgrade, put_subscription_pending_change)
        kind, arg = action
        key = None
        if kind == "upgrade":
            from agent.billing_view import new_idempotency_key
            key = idempotency_key or new_idempotency_key()
        try:
            if kind == "upgrade":
                res = post_subscription_upgrade(subscription_type_id=arg, idempotency_key=key) or {}
                status = res.get("status")
                name = res.get("targetTierName") or t("cli.subscription.your_new_plan")
                if status in _UPGRADE_OK_KEYS:
                    self._ok(t(_UPGRADE_OK_KEYS[status], name=name))
                elif status in _UPGRADE_STATUS_KEYS:
                    line_key, echo_url = _UPGRADE_STATUS_KEYS[status]
                    _cprint(f"  {t(line_key)}")
                    if echo_url and res.get("recoveryUrl"):
                        _cprint(f"  {t('cli.billing.portal_line', url=res.get('recoveryUrl'))}")
                else:  # unknown / absent 2xx status → also ambiguous, not a flat failure
                    self._subscription_render_upgrade_ambiguous(None)
                return
            pending = {
                "schedule": (put_subscription_pending_change, {"subscription_type_id": arg}),
                "cancel": (put_subscription_pending_change, {"cancel": True}),
                "resume": (delete_subscription_pending_change, {})}.get(kind)
            if pending:
                pending[0](**pending[1])
                self._ok(t(_PENDING_OK_KEYS[kind]))
            self._dim(t("cli.subscription.rerun_to_review"))
        except BillingScopeRequired:  # rejects BEFORE charging → route to the step-up
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=action, idempotency_key=key)
            else:
                print(f"  {t('cli.billing.stepup_stale')}")
        except BillingError as exc:
            # Upgrade only: deterministic PRE-charge rejections (Transient/401/403 types, 4xx codes)
            # never reached Stripe → recovery copy. Transport / 5xx is INDETERMINATE (NAS may have
            # charged) → steer to a re-check, never a blind retry (a fresh key can't dedup).
            _pre_charge = (BillingTransient, BillingSessionRevoked, BillingRemoteSpendingRevoked)
            _ambiguous = (exc.error in ("network_error", "endpoint_unavailable")
                          or exc.status is None or exc.status >= 500)
            if kind == "upgrade" and _ambiguous and not isinstance(exc, _pre_charge):
                self._subscription_render_upgrade_ambiguous(exc)
            else:
                self._subscription_render_error(state, exc)

    def _subscription_handle_scope_required(self, state, *, retry, idempotency_key=None):
        """insufficient_scope → step-up, then replay `retry` ONCE so the user never re-runs the command."""
        granted = self._step_up_remote_spending(
            explain=t("cli.subscription.stepup_explain"),
            noninteractive_msg=f"  {t('cli.subscription.stepup_noninteractive')}",
            declined_msg=f"  {t('cli.subscription.stepup_declined')}",
            not_granted_msg=f"  {t('cli.subscription.stepup_not_granted')}")
        if not granted:
            return
        self._ok(t("cli.billing.remote_spending_allowed"))
        # Bust the 30s token cache (it still holds the pre-grant token; _request only busts on 401).
        try:
            from hermes_cli import nous_billing as _nb
            _nb.invalidate_cached_token()
        except Exception:
            pass
        # Re-fetch fresh state, then replay the held action ONCE (allow_stepup=False).
        from agent.subscription_view import build_subscription_state
        try:
            fresh = build_subscription_state()
        except Exception:
            fresh = state
        if retry[0] == "preview":
            self._subscription_preview_and_confirm(fresh, retry[1], allow_stepup=False)
        else:
            self._subscription_apply(fresh, retry, idempotency_key=idempotency_key, allow_stepup=False)

    def _subscription_render_error(self, state, exc):
        """Render a subscription BillingError (a lighter _billing_render_charge_error)."""
        from cli import _cprint
        msg = str(exc) or t("cli.billing.something_went_wrong")
        if exc.error == "insufficient_scope":  # defensive: the flow routes scope to the step-up before here
            _cprint(f"  {t('cli.subscription.remote_spending_not_allowed_yet')}")
        elif exc.error in ("subscription_mutation_rejected", "preview_rejected"):
            _cprint(f"  🟡 {msg}")
        else:
            _cprint(f"  🔴 {msg}")
        self._print_portal_line(exc)

    def _subscription_render_upgrade_ambiguous(self, exc):
        """AMBIGUOUS outcome (NAS may have charged) → steer to a re-check, never a blind retry (key isn't persisted)."""
        from cli import _cprint
        _cprint(f"  {t('cli.subscription.upgrade_ambiguous')}")
        self._dim(t("cli.subscription.rerun_to_check_plan"))
        self._print_portal_line(exc)

    # ── /topup — Remote Spending (CLI surface, all 5 screens) ──

    def _show_billing(self, command: str = "/topup"):
        """`/topup` — ZERO sub-commands (argument ignored; Overview is the only route). Non-interactive never
        prompts. Money is Decimal end-to-end; the terminal never collects card details."""
        from agent.billing_view import build_billing_state
        state = build_billing_state()
        if not state.logged_in:
            self._print_logged_out(state, t("cli.billing.load_failed_label"), "/topup")
            return
        self._billing_overview(state)

    def _billing_portal_hint(self, state, *, reason: str = "") -> None:
        """Print a portal deep-link line (the funnel for portal-only actions)."""
        if not state.portal_url:
            return
        if reason:
            print(f"  {reason}")
        print(f"  {t('cli.billing.manage_on_portal_url', url=state.portal_url)}")

    def _billing_require_admin(self, state, *, icon="💳 ", off_reason_key="cli.billing.killswitch_reason_buy") -> bool:
        """Admin + org kill-switch gate; portal funnel + False when blocked. ``icon`` adds a blank line + prefix."""
        if state.can_change_plan and state.cli_billing_enabled:
            return True
        if icon:
            print()
        if not state.can_change_plan:
            self._dim(t("cli.billing.actions_require_admin"), icon=icon)
            self._billing_portal_hint(state)
        else:
            self._dim(t("cli.billing.remote_spending_off_for_org"), icon=icon)
            self._billing_portal_hint(state, reason=t(off_reason_key))
        return False

    def _billing_overview(self, state):
        """Screen 1 — balance, bars, menu. No scope preflight (a charge 403s); a missing card does NOT gate it."""
        from cli import _cprint, _b
        from agent.billing_view import format_money
        usage = self._try_usage_model()
        print()
        _cprint(f"  💳 {_b(t('cli.billing.topup_header', balance=format_money(state.balance_usd)))}")
        self._print_org_line(state)
        print(f"  {_RULE}")
        for _bar_ln in self._usage_bar_lines(usage, usage.plan_name if usage else None):
            print(_bar_ln)
        ar = state.auto_reload
        if ar is not None:
            if ar.enabled:
                print(f"  {t('cli.billing.auto_reload_on_line', threshold=format_money(ar.threshold_usd), reload_to=format_money(ar.reload_to_usd))}")
            else:
                print(f"  {t('cli.billing.auto_reload_off_line')}")
        if state.can_change_plan and state.cli_billing_enabled:  # card at a glance, full-menu case only
            if state.card is not None:
                print(f"  {t('cli.billing.card_line', card=state.card.display)}")
            else:
                self._dim(t("cli.billing.no_saved_card_add_funds_hint"))
        print(f"  {_RULE}")
        # Action gating: admin + kill-switch for charge/auto-reload; everyone gets portal.
        if not self._billing_require_admin(state, icon="", off_reason_key="cli.billing.killswitch_reason_overview"):
            return
        if not self._app:  # non-interactive: no modal, just the portal funnel
            self._billing_portal_hint(state)
            return
        # One-time vs automatic — the distinction stated up front in each first sentence.
        self._dim(t("cli.billing.add_funds_now_hint"))
        _amounts = [ar.reload_to_usd, ar.threshold_usd] if ar is not None and ar.enabled else [None]
        if all(a is not None and a.is_finite() for a in _amounts):
            _auto_line = t("cli.billing.refill_when_low_amounts", reload_to=format_money(ar.reload_to_usd),
                           threshold=format_money(ar.threshold_usd))
        else:
            _auto_line = t("cli.billing.refill_when_low_generic")
        self._dim(_auto_line)
        print(f"  {_RULE}")
        # No "Allow Remote Spending" item — discovered at pay time. "Add funds" charges the org's
        # portal-saved card (server-held; no card ref leaves the client).
        action = {
            "buy": self._billing_buy_flow,
            "auto": self._billing_auto_reload_flow,
            "limit": self._billing_limit_screen,
            "portal": self._billing_open_portal}.get(self._modal_choice(t("cli.billing.topup_title"), "", _topup_menu_choices()))
        if action:
            action(state)
        else:
            print(f"  {t('cli.billing.cancelled')}")

    def _billing_open_portal(self, state):
        if not state.portal_url:
            print(f"  {t('cli.billing.no_portal_url')}")
            return
        self._open_or_print_url(state.portal_url)
        print(f"  {t('cli.billing.complete_in_browser')}")

    def _billing_add_card_flow(self, state):
        """No card → add it on the portal (never in-terminal), bounded re-check loop. Refreshed state, or None."""
        from cli import _cprint
        self._block_header("💳", t("cli.billing.add_card_first"), rule=False)
        _cprint(f"  {t('cli.billing.no_saved_card')}")
        self._dim(t("cli.billing.add_card_once_hint"))
        for _ in range(8):  # bounded: portal-open plus a handful of re-checks
            choice = self._modal_choice(t("cli.billing.add_card_title"), "", _add_card_choices())
            if choice == "portal":
                self._billing_open_portal(state)
                self._dim(t("cli.billing.add_card_then_recheck"))
            elif choice == "recheck":
                from agent.billing_view import build_billing_state
                try:
                    fresh = build_billing_state()
                except Exception:
                    fresh = None
                if fresh is not None and fresh.logged_in:
                    state = fresh
                if state.card is not None:
                    self._ok(t("cli.billing.card_found_continuing", card=state.card.display))
                    return state
                print(f"  {t('cli.billing.still_no_card')}")
            else:
                break
        print(f"  {t('cli.billing.cancelled_no_funds')}")
        return None

    def _billing_buy_flow(self, state):
        """Screen 2 (presets) → Screen 3 (confirm+charge+poll). No scope preflight: react to the server's 403s."""
        from agent.billing_view import format_money, validate_charge_amount
        if not self._billing_require_admin(state):
            return
        if not self._app:
            self._block_header("💳", t("cli.billing.add_funds"), rule=False)
            print(f"  {t('cli.billing.presets', presets=', '.join(format_money(p) for p in state.charge_presets))}")
            print(f"  {t('cli.billing.run_interactive_to_purchase')}")
            self._billing_portal_hint(state)
            return
        if state.card is None:  # guided add-card path first, so the amount pick can't 403
            state = self._billing_add_card_flow(state)
            if state is None or state.card is None:
                return
        preset_choices = [(str(p), format_money(p), t("cli.billing.desc_one_time_purchase")) for p in state.charge_presets]
        preset_choices.append(("custom", t("cli.billing.choice_custom_amount"), t("cli.billing.desc_custom_amount")))
        preset_choices.append(_cancel_row())
        card = state.card
        choice = self._modal_choice(
            t("cli.billing.add_funds"),
            t("cli.billing.payment_line", card=card.display) if card else t("cli.billing.no_saved_card_short"),
            preset_choices)
        if not choice or choice == "cancel":
            print(f"  {t('cli.billing.cancelled_no_funds')}")
            return
        from decimal import Decimal
        if choice == "custom":
            entered = self._prompt_text_input(f"  {t('cli.billing.amount_prompt')} ")
            if entered is None:  # cancelled (e.g. slash-worker can't prompt off-thread)
                print(f"  {t('cli.billing.cancelled_no_funds')}")
                return
            v = validate_charge_amount(entered or "", min_usd=state.min_usd, max_usd=state.max_usd)
            if not v.ok:
                print(f"  🔴 {v.error}")
                return
            amount = v.amount
        else:
            try:
                amount = Decimal(choice)
            except Exception:
                print(f"  {t('cli.billing.invalid_selection')}")
                return
        self._billing_confirm_and_charge(state, amount)

    def _billing_confirm_and_charge(self, state, amount):
        """Screen 3 — confirm total + consent, charge, then poll to settlement."""
        from agent.billing_view import format_money, new_idempotency_key
        card = state.card
        self._block_header("💳", t("cli.billing.confirm_purchase"))
        print(f"  {t('cli.billing.total_line', amount=format_money(amount))}")
        if card:
            print(f"  {t('cli.billing.payment_line', card=card.display)}")
            if card.provenance is None:  # older NAS without provenance → generic line
                self._dim(t("cli.billing.portal_card_will_be_charged"))
        print(f"  {_RULE}")
        self._dim(t("cli.billing.consent_charge"))
        confirm_choices = [
            ("pay", t("cli.billing.choice_pay_now", amount=format_money(amount)), t("cli.billing.desc_submit_charge")),
            ("portal", t("cli.billing.choice_manage_on_portal"), t("cli.billing.desc_manage_card_billing")),
            _cancel_row(label_key="cli.billing.choice_go_back", desc_key="cli.billing.desc_do_not_charge")]
        if not self._app:
            print(f"  {t('cli.billing.run_interactive_to_confirm')}")
            return
        choice = self._modal_choice(
            t("cli.billing.pay_question", amount=format_money(amount)),
            card.display if card else t("cli.billing.no_saved_card_lower"), confirm_choices)
        if choice == "portal":
            self._billing_open_portal(state)
            return
        if choice != "pay":
            print(f"  {t('cli.billing.cancelled_no_funds')}")
            return
        key = new_idempotency_key()  # reused on the post-step-up resume so a double-submit collapses
        self._billing_submit_and_poll(
            state, amount, key, missing_msg=f"  🔴 {t('cli.billing.no_charge_id')}",
            status_msg=t("cli.billing.charge_submitted"),
            on_scope=lambda: self._billing_handle_scope_required(state, amount=amount, idempotency_key=key))

    def _billing_submit_and_poll(self, state, amount, key, *, missing_msg, status_msg, on_scope=None):
        """POST the charge, then poll. ``on_scope`` handles a scope denial (first submit); else it renders."""
        from cli import _cprint, _d
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, post_charge
        try:
            result = post_charge(amount_usd=amount, idempotency_key=key)
        except BillingError as exc:
            if on_scope is not None and isinstance(exc, BillingScopeRequired):
                on_scope()
            else:
                self._billing_render_charge_error(state, exc)
            return
        charge_id = result.get("chargeId")
        if not charge_id:
            print(missing_msg)
            return
        _cprint(f"  {_d(status_msg)}")
        self._billing_poll_charge(state, charge_id, amount)

    def _billing_poll_charge(self, state, charge_id, amount):
        """Poll loop: 2s interval, 5-min cap, cancellable. settled = ledger truth."""
        import time as _time
        from agent.billing_view import format_money, parse_money
        from hermes_cli.nous_billing import BillingError, BillingTransient, get_charge_status
        deadline = _time.time() + 300
        while _time.time() < deadline:
            try:
                status = get_charge_status(charge_id)
            except BillingTransient as exc:  # retry-after, NOT a failure — back off and keep polling
                _time.sleep(min(exc.retry_after or 5, 30))
                continue
            except BillingError as exc:
                print(f"  {t('cli.billing.could_not_check_charge', error=exc)}")
                return
            state_str = status.get("status")
            if state_str == "settled":
                amt = status.get("amountUsd")
                _added = format_money(parse_money(amt)) if amt else format_money(amount)
                print(f"  {t('cli.billing.added_to_balance', amount=_added)}")
                return
            if state_str == "failed":
                self._billing_render_charge_failed(state, status.get("reason"))
                return
            _time.sleep(2.0)  # pending
        print(f"  {t('cli.billing.still_processing')}")
        self._billing_portal_hint(state)

    def _billing_render_charge_failed(self, state, reason):
        """Poll `failed` reasons → the right copy + portal funnel."""
        reason = (reason or "").strip()
        key = _CHARGE_FAILED_KEYS.get(reason)
        print(f"  {t(key)}" if key else f"  {t('cli.billing.charge_failed_generic', reason=reason or 'processing_error')}")
        self._billing_portal_hint(state)

    def _billing_render_charge_error(self, state, exc):
        """Submit-time BillingError. Order matters: revoked/session before code lookups; Transient before scope."""
        from hermes_cli.nous_billing import BillingTransient, BillingRemoteSpendingRevoked, BillingSessionRevoked
        code = exc.error
        portal_url = exc.portal_url or state.portal_url
        if isinstance(exc, BillingRemoteSpendingRevoked) or code == "remote_spending_revoked":
            # This terminal's spend was revoked; recovery is reconnect.
            who = t("cli.billing.revoked_by_admin") if exc.actor == "admin" else t("cli.billing.revoked_by_you")
            print(f"  {t('cli.billing.revoked_reconnect', who=who)}")
        elif isinstance(exc, BillingSessionRevoked) or code == "session_revoked":
            print(f"  {t('cli.billing.session_logged_out')}")
        elif code in _CHARGE_ERROR_KEYS or exc.code == "remote_spending_disabled":
            # Fixed copy by `error`; the gate's dual error/code payload may carry it in `.code` only.
            print(f"  {t(_CHARGE_ERROR_KEYS.get(code) or _CHARGE_ERROR_KEYS['cli_billing_disabled'])}")
        elif code == "monthly_cap_exceeded":
            remaining = (exc.payload or {}).get("remainingUsd")
            if remaining is not None:
                print(f"  {t('cli.billing.monthly_cap_reached_headroom', remaining=remaining)}")
            else:
                print(f"  {t('cli.billing.monthly_cap_reached')}")
        elif isinstance(exc, BillingTransient):
            wait = exc.retry_after
            mins = t("cli.billing.try_again_in_min", minutes=max(1, round(wait / 60))) if wait else ""
            print(f"  {t('cli.billing.too_many_charges', mins=mins)}")
        elif code == "insufficient_scope":
            # Never leak the raw billing:manage scope (a raced post-grant replay can re-raise it).
            print(f"  {t('cli.billing.remote_spending_needs_approval')}")
        else:
            print(f"  🔴 {exc}")
        if portal_url:
            print(f"  {t('cli.billing.portal_line', url=portal_url)}")

    def _billing_handle_scope_required(self, state, *, amount=None, idempotency_key=None):
        """403 insufficient_scope → reauth, then resume ``amount`` on explicit confirm, reusing the idempotency key."""
        from agent.billing_view import build_billing_state, format_money, new_idempotency_key
        amount_str = format_money(amount) if amount is not None else t("cli.billing.your_topup")
        granted = self._step_up_remote_spending(
            explain=t("cli.billing.charge_stepup_explain", amount=amount_str),
            noninteractive_msg=f"  {t('cli.billing.charge_stepup_noninteractive')}",
            declined_msg=f"  {t('cli.billing.charge_stepup_declined')}",
            not_granted_msg=f"  {t('cli.billing.charge_stepup_not_granted')}")
        if not granted:
            return
        # The token now has the scope, but the ORG kill-switch is a separate gate — re-fetch /state.
        fresh = build_billing_state()
        if not (fresh.logged_in and fresh.cli_billing_enabled):
            print(f"  {t('cli.billing.allowed_but_org_off')}")
            self._billing_portal_hint(fresh)
            return
        if fresh.card is None:  # half-done state: say so rather than a bare "✓ enabled"
            print(f"  ✓ {t('cli.billing.allowed_no_card')}")
            self._dim(t("cli.billing.topup_on_portal_to_continue"))
            self._billing_portal_hint(fresh)
            return
        if amount is None:  # scope-required hit outside a charge (e.g. auto-reload config)
            print(f"  ✓ {t('cli.billing.allowed_run_topup')}")
            return
        print(f"  ✓ {t('cli.billing.remote_spending_allowed')}")
        resume_choices = [
            ("resume", t("cli.billing.choice_resume_topup", amount=format_money(amount)), t("cli.billing.desc_resume_topup")),
            _cancel_row(desc_key="cli.billing.desc_do_not_charge")]
        if self._modal_choice(
                t("cli.billing.resume_topup_title"),
                t("cli.billing.resume_topup_detail", amount=format_money(amount)), resume_choices) != "resume":
            print(f"  {t('cli.billing.cancelled_no_funds')}")
            return
        self._billing_submit_and_poll(
            fresh, amount, idempotency_key or new_idempotency_key(),
            missing_msg=f"  {t('cli.billing.no_charge_id')}",
            status_msg=t("cli.billing.resuming_topup"))

    def _billing_auto_reload_flow(self, state):
        """Screen 4 — threshold + reload-to → PATCH. Prefills; validates ``reload_to > threshold``; "Turn off" if on."""
        from agent.billing_view import format_money, validate_charge_amount
        if not self._billing_require_admin(state):
            return
        card = state.card
        ar = state.auto_reload
        currently_on = bool(ar and ar.enabled)
        self._block_header("💳", t("cli.billing.auto_reload"))
        self._dim(t("cli.billing.auto_reload_hint"))
        if card:
            print(f"  {t('cli.billing.card_on_file', card=card.masked)}")
        else:
            print(f"  {t('cli.billing.no_saved_card_manage_portal')}")
            self._billing_portal_hint(state)
            return
        _current = (t("cli.billing.auto_reload_rule", threshold=format_money(ar.threshold_usd), reload_to=format_money(ar.reload_to_usd))
                    if currently_on else "")
        if currently_on:
            print(f"  {t('cli.billing.currently', rule=_current)}")
        if not self._app:
            print(f"  {t('cli.billing.run_interactive_to_configure')}")
            self._billing_portal_hint(state)
            return
        if currently_on:  # let the user turn it off without re-entering values
            top = self._modal_choice(t("cli.billing.auto_reload"), t("cli.billing.auto_reload_on_detail", rule=_current),
                                     _auto_reload_top_choices())
            if top == "off":
                self._billing_auto_reload_disable(state)
                return
            if top != "edit":
                print(f"  {t('cli.billing.cancelled_yellow')}")
                return
        _CANCELLED = object()

        def _ask_amount(label, current):
            """One amount; empty keeps `current` when editing. Decimal / kept value, or _CANCELLED (already printed)."""
            cur = format_money(current) if currently_on else None
            prompt = f"  {t('cli.billing.amount_prompt_labeled', label=label)}" + (f" [{cur}]: " if cur else ": ")
            raw = self._prompt_text_input(prompt)
            if raw is None:  # cancelled (e.g. slash-worker can't prompt off-thread)
                print(f"  {t('cli.billing.cancelled_yellow')}")
                return _CANCELLED
            if not (raw or "").strip() and currently_on:
                return current
            v = validate_charge_amount(raw or "", min_usd=state.min_usd, max_usd=state.max_usd)
            if not v.ok or v.amount is None:
                print(f"  🔴 {v.error}")
                return _CANCELLED
            return v.amount
        threshold_amt = _ask_amount(t("cli.billing.threshold_label"), ar.threshold_usd if currently_on else None)
        if threshold_amt is _CANCELLED:
            return
        reload_amt = _ask_amount(t("cli.billing.reload_to_label"), ar.reload_to_usd if currently_on else None)
        if reload_amt is _CANCELLED:
            return
        if reload_amt is None or threshold_amt is None or reload_amt <= threshold_amt:
            print(f"  {t('cli.billing.reload_must_exceed_threshold')}")
            return
        self._dim(t("cli.billing.auto_reload_consent", card=card.masked, threshold=format_money(threshold_amt)), lead=True)
        if self._modal_choice(
                t("cli.billing.turn_on_auto_reload_question"),
                t("cli.billing.auto_reload_rule_cap", threshold=format_money(threshold_amt), reload_to=format_money(reload_amt)),
                _auto_reload_agree_choices()) != "agree":
            print(f"  {t('cli.billing.cancelled_yellow')}")
            return
        if self._billing_patch_auto_top_up(state, enabled=True, threshold=float(threshold_amt), top_up_amount=float(reload_amt)):
            print(f"  {t('cli.billing.auto_reload_enabled', threshold=format_money(threshold_amt), reload_to=format_money(reload_amt))}")

    def _billing_patch_auto_top_up(self, state, **kwargs) -> bool:
        """PATCH auto-top-up; scope denials → step-up, other errors → renderer. True on success."""
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, patch_auto_top_up
        try:
            patch_auto_top_up(**kwargs)
        except BillingScopeRequired:
            self._billing_handle_scope_required(state)
            return False
        except BillingError as exc:
            self._billing_render_charge_error(state, exc)
            return False
        return True

    def _billing_auto_reload_disable(self, state):
        """PATCH ``enabled:false``; the endpoint still requires threshold/topUpAmount → echo current (or 0)."""
        ar = state.auto_reload
        thr = float(ar.threshold_usd) if ar and ar.threshold_usd is not None else 0.0
        rel = float(ar.reload_to_usd) if ar and ar.reload_to_usd is not None else 0.0
        if self._billing_patch_auto_top_up(state, enabled=False, threshold=thr, top_up_amount=rel):
            print(f"  {t('cli.billing.auto_reload_disabled')}")

    def _billing_limit_screen(self, state):
        """Screen 5 — monthly spend limit (read-only; cap is portal-only)."""
        from agent.billing_view import format_money
        self._block_header("💳", t("cli.billing.monthly_spend_limit"))
        cap = state.monthly_cap
        if cap is None or cap.limit_usd is None:
            self._dim(t("cli.billing.no_monthly_cap"))
        else:
            ceiling = t("cli.billing.default_ceiling_suffix") if cap.is_default_ceiling else ""
            print(f"  {t('cli.billing.used_this_month', spent=format_money(cap.spent_this_month_usd), limit=format_money(cap.limit_usd), ceiling=ceiling)}")
        self._dim(t("cli.billing.monthly_limit_read_only"))
        self._billing_portal_hint(state)
