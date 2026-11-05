# React Design Conventions & Enterprise FOSS Stack

React is unopinionated about styling, which means design discipline must come from the design system and component architecture, not the framework.

## Enterprise FOSS Component Stack (Prime Ecosystem & Tailwind/Bootstrap)

When building enterprise React applications, leverage 100% Free and Open Source (FOSS) component libraries (MIT License):

- **PrimeReact**: The direct React counterpart to PrimeNG (by PrimeTek), featuring rich data tables, dialogs, calendars, and dropdowns sharing the same Lara/Aura theme ecosystem.
- **Shadcn/ui & Tailwind CSS**: Accessible, copy-pasteable component primitives styled with utility-first Tailwind CSS.
- **Bootstrap 5 & React-Bootstrap**: For projects utilizing standard Bootstrap utility classes and grid layout systems.
- **TanStack Table (React Table)**: Headless table library for high-performance data grids with sorting, filtering, and pagination.
- **React Hook Form + Zod**: Performant form management with type-safe validation schemas.

## Component Boundaries & Styling Strategy

- **Composition over configuration**: Small, composable components with clear props (`variant`, `size`).
- **Single source of truth for tokens**: Configure Tailwind config, Bootstrap CSS variables, or CSS custom properties for color scales and spacing (8px grid).

## UI Pattern Reference & Examples

### 1. Buttons & Action States (PrimeReact vs Shadcn/Bootstrap)
- **PrimeReact Button**:
  ```tsx
  import { Button } from 'primereact/button';

  export function ActionButton({ loading }: { loading: boolean }) {
      return <Button label="Save" icon="pi pi-check" loading={loading} />;
  }
  ```
- **Bootstrap / React-Bootstrap Button**:
  ```tsx
  import Button from 'react-bootstrap/Button';
  import Spinner from 'react-bootstrap/Spinner';

  export function BootstrapButton({ loading }: { loading: boolean }) {
      return (
          <Button variant="primary" disabled={loading}>
              {loading && <Spinner animation="border" size="sm" className="me-2" />}
              Save Changes
          </Button>
      );
  }
  ```

### 2. Combos, Selects & Dropdowns
- **PrimeReact Dropdown**:
  ```tsx
  import { Dropdown } from 'primereact/dropdown';

  export function EditionDropdown({ selectedEdition, onSelect, editions }) {
      return (
          <Dropdown value={selectedEdition} options={editions} optionLabel="name" 
                    onChange={(e) => onSelect(e.value)} placeholder="Select Edition" className="w-full md:w-14rem" />
      );
  }
  ```

### 3. Date & Time Pickers
- **PrimeReact Calendar**:
  ```tsx
  import { Calendar } from 'primereact/calendar';

  export function DatePickerField({ date, setDate }) {
      return <Calendar value={date} onChange={(e) => setDate(e.value)} showTime dateFormat="dd/mm/yy" />;
  }
  ```

### 4. Enterprise Data Tables
- **PrimeReact DataTable**:
  ```tsx
  import { DataTable } from 'primereact/datatable';
  import { Column } from 'primereact/column';

  export function EnterpriseTable({ customers, loading }) {
      return (
          <DataTable value={customers} paginator rows={10} loading={loading} className="p-datatable-gridlines">
              <Column field="name" header="Name" sortable></Column>
              <Column field="email" header="Email"></Column>
          </DataTable>
      );
  }
  ```

### 5. Modals & Dialogs
- **PrimeReact Dialog / Bootstrap Modal**: Accessible dialogs with structured header, body, and action footer.

## Responsive Patterns & Common Traps

- Use CSS Grid and Flexbox for page and component layouts. Ensure touch targets are ≥ 44x44px. Avoid inline styles and let design tokens drive the visual identity.
