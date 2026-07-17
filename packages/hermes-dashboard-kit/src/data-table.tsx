import { useMemo, useState, type ReactNode } from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { ChevronDown, ChevronUp, Search } from "lucide-react";
import { cn } from "./utils";
import { DashboardEmptyState, DashboardErrorState, DashboardLoadingState } from "./states";

export interface DataTableColumn<T> {
  id: string;
  header: ReactNode;
  accessor?: (row: T) => ReactNode;
  sortValue?: (row: T) => string | number | boolean | null | undefined;
  className?: string;
  cellClassName?: string;
}

export function DataTable<T>({
  rows,
  columns,
  getRowKey,
  loading = false,
  error,
  emptyTitle,
  emptyDescription,
  onRowClick,
  className,
}: {
  rows: T[];
  columns: DataTableColumn<T>[];
  getRowKey: (row: T, index: number) => string;
  loading?: boolean;
  error?: string;
  emptyTitle?: string;
  emptyDescription?: string;
  onRowClick?: (row: T) => void;
  className?: string;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const sourceColumns = useMemo(() => new Map(columns.map((column) => [column.id, column])), [columns]);
  const tableColumns = useMemo<ColumnDef<T>[]>(() => columns.map((column) => ({
    id: column.id,
    accessorFn: column.sortValue ? (row) => column.sortValue?.(row) ?? "" : () => "",
    cell: (info) => column.accessor ? column.accessor(info.row.original) : String(info.getValue() ?? ""),
    enableSorting: Boolean(column.sortValue),
    header: () => column.header,
    sortingFn: "alphanumeric",
  })), [columns]);
  const table = useReactTable({
    columns: tableColumns,
    data: rows,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row, index) => getRowKey(row, index),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: { sorting },
  });

  if (loading) return <DashboardLoadingState label="Loading table" />;
  if (error) return <DashboardErrorState message={error} />;
  if (!rows.length) return <DashboardEmptyState title={emptyTitle} description={emptyDescription} />;

  return (
    <div className={cn("overflow-hidden rounded-lg border border-border bg-card", className)}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[42rem] border-collapse text-sm">
          <thead className="bg-muted/60 text-left text-xs uppercase tracking-wide text-muted-foreground">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const sourceColumn = sourceColumns.get(header.column.id);
                  const sorted = header.column.getIsSorted();
                  const Icon = sorted === "desc" ? ChevronDown : ChevronUp;
                  return (
                    <th key={header.id} className={cn("border-b border-border px-3 py-2 font-medium", sourceColumn?.className)}>
                      {header.column.getCanSort() ? (
                        <button
                          className="inline-flex items-center gap-1 text-left"
                          onClick={header.column.getToggleSortingHandler()}
                          type="button"
                        >
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          <Icon className={cn("h-3.5 w-3.5", sorted ? "opacity-100" : "opacity-30")} aria-hidden="true" />
                        </button>
                      ) : (
                        flexRender(header.column.columnDef.header, header.getContext())
                      )}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className={cn("border-b border-border last:border-b-0", onRowClick && "cursor-pointer hover:bg-muted/50")}
                onClick={() => onRowClick?.(row.original)}
              >
                {row.getVisibleCells().map((cell) => {
                  const sourceColumn = sourceColumns.get(cell.column.id);
                  return (
                    <td key={cell.id} className={cn("px-3 py-2 align-top", sourceColumn?.cellClassName)}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function TableToolbar({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={cn("flex flex-wrap items-center justify-between gap-3", className)}>{children}</div>;
}

export function FilterBar({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={cn("flex flex-wrap items-center gap-2 rounded-lg border border-border bg-card p-2", className)}>{children}</div>;
}

export function SearchInput({
  value,
  onChange,
  placeholder = "Search",
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}) {
  return (
    <label className={cn("relative block min-w-56 flex-1", className)}>
      <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
      <input
        className="h-9 w-full rounded-md border border-border bg-background px-9 text-sm text-foreground outline-none transition focus:border-primary"
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        type="search"
        value={value}
      />
    </label>
  );
}

export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  className,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  className?: string;
}) {
  return (
    <div className={cn("inline-flex rounded-md border border-border bg-muted p-1", className)}>
      {options.map((option) => (
        <button
          key={option.value}
          className={cn(
            "h-7 rounded px-3 text-xs font-medium transition",
            option.value === value ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
          onClick={() => onChange(option.value)}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function DateRangeToggle({
  value,
  onChange,
}: {
  value: "7d" | "30d" | "90d";
  onChange: (value: "7d" | "30d" | "90d") => void;
}) {
  return (
    <SegmentedControl
      value={value}
      onChange={onChange}
      options={[
        { value: "7d", label: "7 days" },
        { value: "30d", label: "30 days" },
        { value: "90d", label: "90 days" },
      ]}
    />
  );
}
