# SZ Stock Filter Enum Options

These values are exact SDK enum strings to use when composing `stock_filter_specs`.

## `sort_dir`

- `ASCEND`
- `DESCEND`

## `financial_quarter`

- `ANNUAL`
- `FIRST_QUARTER`
- `INTERIM`
- `THIRD_QUARTER`
- `MOST_RECENT_QUARTER`

## `supported_pattern_ktype`

- `K_60M`
- `K_DAY`
- `K_WEEK`
- `K_MON`

## `relative_position`

- `MORE`
- `LESS`
- `CROSS_UP`
- `CROSS_DOWN`

## `stock_filter_spec_shape`

```json
{
  "type": "simple | accumulate | financial | pattern | custom_indicator",
  "stock_field": "one of filter_types[type].fields",
  "filter_min": "optional number",
  "filter_max": "optional number",
  "sort": "optional ASCEND | DESCEND",
  "days": "accumulate only; default 1",
  "quarter": "financial only; default ANNUAL",
  "ktype": "pattern/custom_indicator only; default K_DAY"
}
```
