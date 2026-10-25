# Angular Design Conventions & Enterprise FOSS Stack

Angular favors a more structured, opinionated component model. The design should respect the framework's separation of concerns without being dictated by it.

## Enterprise FOSS Component Stack

When designing enterprise Angular applications (such as middleware portals and admin dashboards), rely on 100% Free and Open Source (FOSS) libraries under permissive licenses (MIT/Apache 2.0):

- **PrimeNG (^17.x)**: For advanced data tables (`p-table`), dialogs (`p-dialog`), calendars (`p-calendar`), and tree/menu structures.
- **`@ng-select/ng-select` (^12.x)**: For searchable comboboxes, single/multi-select dropdowns, and async data loading.
- **ngx-bootstrap (^12.x)**: For datepickers (`bsDatepicker`), tooltips, and modal management.
- **Bootstrap 5 & Bootstrap Icons**: For responsive grid systems, flexbox layouts, spacing utilities, and iconography.
- **SweetAlert2 & FreezeUI**: For non-blocking notifications, confirmations, and async loading states.

## Component Boundaries & Styling Strategy

- **One component per responsibility**: Split page shells, feature modules, and shared UI into distinct layers.
- **Component-scoped styles**: Prefer `ViewEncapsulation.Emulated` with CSS custom properties for design tokens.
- **Global styles**: Define type scales, color palettes, and 8px spacing grids in global stylesheets. Avoid commercial proprietary themes (e.g., legacy Metronic bundles) in new designs.

## UI Pattern Reference

### 1. Buttons & Action States
- Use Bootstrap 5 classes: `btn btn-primary`, `btn btn-secondary`, `btn btn-success`, `btn btn-danger`, `btn btn-info`.
- For form submissions and async actions, incorporate busy state indicators (`[buttonBusy]`, disabling interaction while processing).
- For table row actions, use clean icon buttons: `btn btn-sm btn-clean btn-icon btn-icon-md`.

### 2. Combos, Selects & Dropdowns
- **Simple Select**: `<select class="form-control">` with native options.
- **Advanced / Searchable Select**: `<ng-select [items]="items" bindLabel="name" bindValue="id" [(ngModel)]="selectedId"></ng-select>` for rich search and multi-select.
- **Dropdown Actions**: ngx-bootstrap dropdown directives (`dropdown`, `dropdownToggle`, `dropdownMenu`).

### 3. Date & Time Pickers
- **ngx-bootstrap Datepicker**: `<input class="form-control" bsDatepicker [bsConfig]="{ containerClass: 'theme-dark-blue' }" />` for intuitive date selection.
- **PrimeNG Calendar**: `<p-calendar [(ngModel)]="date" [showTime]="true"></p-calendar>` for timestamp and range filtering.

### 4. Enterprise Data Tables
- Use PrimeNG `p-table` with lazy loading (`[lazy]="true"`), pagination, sorting, and responsive wrappers (`div.table-responsive`).
- Maintain clear visual hierarchy: alternating row colors, clear table headers, and accessible column widths.

### 5. Modals & Dialogs
- Standardized modal layout:
  - Header: Title + close button (`×`).
  - Body: Form controls with proper vertical rhythm and spacing.
  - Footer: Cancel (`btn btn-secondary`) and Save (`btn btn-primary` with loading state).

## Responsive Patterns & Accessibility

- Use `BreakpointObserver` from `@angular/cdk/layout` for programmatic breakpoint handling.
- Ensure all interactive elements meet touch target minimums (≥ 44x44px).
- Maintain WCAG AA color contrast across light and dark modes.

## Common Traps

- Overusing framework default palettes without custom theming or brand alignment.
- Storing design tokens in component files instead of global token definitions.
- Creating monolithic templates with deeply nested `*ngIf` statements instead of modular sub-components.
