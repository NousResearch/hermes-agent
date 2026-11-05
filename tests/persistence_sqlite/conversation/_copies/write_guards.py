from . import Copy

COPY = Copy(
    legacy='hermes_state_maintenance',
    doc='Active turn-lease and compression-lock guards on session rows.',
    legacy_class='SessionMaintenanceMixin',
    members=(
        '_guarded_ids',
        '_write_guards_reject',
    ),
)
