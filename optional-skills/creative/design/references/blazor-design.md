# Blazor Design Conventions & Enterprise FOSS Stack

Blazor sits between a traditional server-rendered app and a modern SPA. The design must work for both WebAssembly and Server modes, and it should lean on the component model for consistency.

## Enterprise FOSS Component Stack (Radzen, MudBlazor & Bootstrap Blazor)

When building enterprise Blazor applications, leverage 100% Free and Open Source (FOSS) component libraries (MIT License) that mirror the Prime/Bootstrap enterprise pattern:

- **Radzen Blazor Components**: Comprehensive UI component library (DataGrids, Dropdowns, Calendars, Dialogs) with clean material/fluent styling.
- **MudBlazor**: Rich Material Design component library for Blazor with fast client-side responsiveness.
- **Bootstrap Blazor / BlazorStrap**: Component libraries wrapping native Bootstrap 5 classes and utility components.

## Component Boundaries & Styling Strategy

- **Component per responsibility**: Separate layout (`MainLayout.razor`), pages (`Pages/`), and reusable UI (`Shared/`).
- **Scoped CSS**: Use `.razor.css` for component encapsulation combined with `:root` CSS custom properties and Bootstrap 5 utility classes.

## UI Pattern Reference & Examples (Radzen & Bootstrap Blazor)

### 1. Buttons & Action States
- **Radzen Button with Loading State**:
  ```razor
  <RadzenButton ButtonType="ButtonType.Submit" IsBusy="@isSaving" BusyText="Saving..." Text="Save Changes" ButtonStyle="ButtonStyle.Primary" />
  ```
- **Bootstrap Blazor Button**:
  ```razor
  <Button Color="Color.Primary" IsAsync="true" OnClick="SaveAsync">Save Changes</Button>
  ```

### 2. Combos, Selects & Dropdowns
- **Radzen Dropdown**:
  ```razor
  <RadzenDropDown @-bind-Value="selectedEditionId" Data=@editions TextProperty="Name" ValueProperty="Id" Placeholder="Select Edition" Class="w-100" />
  ```

### 3. Date & Time Pickers
- **Radzen DatePicker**:
  ```razor
  <RadzenDatePicker @-bind-Value=@selectedDate ShowTime="true" DateFormat="dd/MM/yyyy HH:mm" Class="w-100" />
  ```

### 4. Enterprise Data Tables
- **Radzen DataGrid (Server-side paging/sorting)**:
  ```razor
  <RadzenDataGrid Data=@editions TItem="EditionDto" AllowPaging="true" AllowSorting="true" PageSize="10">
      <Columns>
          <RadzenDataGridColumn TItem="EditionDto" Property="Name" Title="Name" />
          <RadzenDataGridColumn TItem="EditionDto" Property="DisplayName" Title="Display Name" />
      </Columns>
  </RadzenDataGrid>
  ```

### 5. Modals & Dialogs
- Use Radzen Dialog service (`DialogService.OpenAsync<CreateEditionModal>`) for modal workflows.

## Responsive Patterns & Common Traps

- Use CSS Grid, Flexbox, and Bootstrap 5 responsive utility classes (`d-flex`, `flex-column`, `gap-3`). Account for Blazor Server latency by prioritizing clear loading feedback.
