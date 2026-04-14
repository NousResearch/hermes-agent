---
type: component_spec
id: component.order_book.v1
status: proposed
trust_level: draft
created: 2026-03-08
updated: 2026-03-08
component: OrderBook
tokens:
  - color.surface.panel
  - color.state.up
  - color.state.down
  - spacing.row.condensed
gate_requirements:
  - G1_SCHEMA
  - G2_STRUCTURE
  - G3_A11Y
  - G4_VISUAL
  - G5_RUNTIME
  - G6_ALIGNMENT
a11y_requirements:
  required_roles:
    - grid
    - row
    - gridcell
  required_aria_attributes:
    - aria-label
    - aria-rowcount
  keyboard_contract:
    - ArrowUp/ArrowDown moves active row
    - Enter opens row detail drawer
---

# Component Spec — OrderBook

## Allowed props
- density_mode: clean_guided | power_user_density
- row_count: integer (min 1)
- show_spread: boolean

## Variant states
- loading
- stale
- read_only
- error
- as_of

## Slot contract
- header
- body
- footer

## Interaction contract refs
- 03_Contracts/orderbook.keyboard.yaml
- 03_Contracts/orderbook.focus.yaml
