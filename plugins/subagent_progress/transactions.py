"""Keep runtime lease publication atomic with the SQL commit."""
from contextlib import contextmanager


@contextmanager
def lease_transaction(lease):
    if lease is None:
        yield
        return
    # The lock spans SQL __exit__ (including COMMIT), not just mutations.
    with lease.lock:
        fields = ('deadline', 'pending', 'latest', 'closed', 'renewals')
        before = {field: getattr(lease, field) for field in fields}
        try:
            yield
        except BaseException:
            for field, value in before.items():
                setattr(lease, field, value)
            raise
