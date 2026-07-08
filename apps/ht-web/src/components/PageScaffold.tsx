import { useCallback, useEffect, useState, type ReactNode } from "react";
import { ApiError } from "@/api/client";

// Shared shell for a REST-backed management page: title bar, optional action
// slot, and consistent loading/error/empty states. Pages call useResource for
// the fetch lifecycle and render into `children`.
export function ManagementPage({
  title,
  actions,
  children,
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="ht-page">
      <header className="ht-page__head">
        <h1 className="ht-page__title">{title}</h1>
        {actions && <div className="ht-page__actions">{actions}</div>}
      </header>
      <div className="ht-page__body">{children}</div>
    </div>
  );
}

export interface Resource<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/** Load a resource once on mount (and on demand via reload). */
export function useResource<T>(loader: () => Promise<T>, deps: unknown[] = []): Resource<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    loader()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.message : String(e));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  const [nonce, setNonce] = useState(0);
  useEffect(() => run(), [run, nonce]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);

  return { data, loading, error, reload };
}

/** Render a resource's loading/error/empty/success states uniformly. */
export function ResourceView<T>({
  resource,
  empty,
  children,
}: {
  resource: Resource<T>;
  empty?: (data: T) => boolean;
  children: (data: T) => ReactNode;
}) {
  if (resource.loading && resource.data === null) {
    return <p className="ht-muted">Loading…</p>;
  }
  if (resource.error) {
    return (
      <div className="ht-error">
        {resource.error}
        <button type="button" className="ht-btn ht-btn--sm" onClick={resource.reload}>
          Retry
        </button>
      </div>
    );
  }
  if (resource.data === null || (empty && empty(resource.data))) {
    return <p className="ht-muted">Nothing to show.</p>;
  }
  return <>{children(resource.data)}</>;
}
