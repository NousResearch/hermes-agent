"""Tests for the product-price-monitor skill's price-watch blueprint."""


def test_price_watch_blueprint_schedule_resolves():
    from cron.blueprint_catalog import CATALOG

    bp = next(b for b in CATALOG if b.key == "price-watch")
    interval_slot = next(s for s in bp.slots if s.name == "interval_h")
    for opt in interval_slot.options:
        expr = bp.schedule_template.format(interval_h=opt)
        fields = expr.split()
        assert len(fields) == 5, f"invalid cron expr: {expr}"
        assert fields[1] == f"*/{opt}"
