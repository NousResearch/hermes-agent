"""Writer-process commit wrapper; only its plain-data facade calls cross RPC."""
from contextlib import contextmanager

from hermes_cli.skill_catalog.locks import writer_hold
from hermes_cli.skill_catalog.projection import _target
from hermes_cli.skill_catalog.validation import CatalogError


@contextmanager
def commit(facade, home, admission, operation_id, acquired):
    """Hold the writer's descriptors, check pre-state, then validate with its parent.

    The provider must check its opaque content revision inside the yielded block
    before writing. Core checks path identity/presence, not provider content hashes.
    """
    try:
        with writer_hold(home, [acquired['target_relative_path']]):
            target = _target(home, acquired['target_relative_path'])
            if target.exists() != (acquired['expected_installation_revision'] is not None):
                raise CatalogError('review_changed', 'Catalog target changed after review.')
            facade.validate(dict(permit_id=acquired['permit_id']))
            yield Commit(facade, acquired, admission, operation_id)
    finally:
        facade.release(dict(permit_id=acquired['permit_id']))


class Commit:
    def __init__(self, facade, acquired, admission, operation_id):
        self._facade = facade
        self._acquired = acquired
        self._admission = admission
        self._operation_id = operation_id
        self.target_relative_path = acquired['target_relative_path']
        self.expected_installation_revision = acquired['expected_installation_revision']

    def settle(self, result):
        return self._facade.settle(dict(permit_id=self._acquired['permit_id'],
            reservation_id=self._admission['reservation_id'], operation_id=self._operation_id,
            expected_path_revision=self._admission['expected_path_revision'], result=result))
