import { type ReactNode } from "react";
export interface DataTableColumn<T> {
    id: string;
    header: ReactNode;
    accessor?: (row: T) => ReactNode;
    sortValue?: (row: T) => string | number | boolean | null | undefined;
    className?: string;
    cellClassName?: string;
}
export declare function DataTable<T>({ rows, columns, getRowKey, loading, error, emptyTitle, emptyDescription, onRowClick, className, }: {
    rows: T[];
    columns: DataTableColumn<T>[];
    getRowKey: (row: T, index: number) => string;
    loading?: boolean;
    error?: string;
    emptyTitle?: string;
    emptyDescription?: string;
    onRowClick?: (row: T) => void;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function TableToolbar({ children, className, }: {
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function FilterBar({ children, className, }: {
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function SearchInput({ value, onChange, placeholder, className, }: {
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function SegmentedControl<T extends string>({ value, options, onChange, className, }: {
    value: T;
    options: {
        value: T;
        label: string;
    }[];
    onChange: (value: T) => void;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function DateRangeToggle({ value, onChange, }: {
    value: "7d" | "30d" | "90d";
    onChange: (value: "7d" | "30d" | "90d") => void;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=data-table.d.ts.map