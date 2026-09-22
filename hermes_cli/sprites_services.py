"""Per-profile gateways supervised by Sprites Services, including sticky stop."""
from __future__ import annotations

from hermes_cli import sprites_api


class SpritesServiceManager:
    kind = 'sprites'

    @staticmethod
    def _profile(name: str) -> str:
        from hermes_cli.service_manager import validate_profile_name
        if not name.startswith('gateway-'):
            raise ValueError('Not a Hermes gateway service')
        profile = name[8:]
        validate_profile_name(profile)
        return profile

    def start(self, name: str) -> None:
        from hermes_cli.service_manager import GatewayNotRegisteredError, _profile_dir_for_gateway_service, _write_gateway_desired_state
        profile = self._profile(name)
        _write_gateway_desired_state(name, 'running')
        try:
            sprites_api.request('POST', f'/services/{name}/start')
        except FileNotFoundError:
            if not _profile_dir_for_gateway_service(name).is_dir():
                raise GatewayNotRegisteredError(name) from None
            self.register_profile_gateway(profile)

    def stop(self, name: str) -> None:
        from hermes_cli.service_manager import _write_gateway_desired_state
        self._profile(name)
        _write_gateway_desired_state(name, 'stopped')
        try:
            sprites_api.request('POST', f'/services/{name}/stop')
        except FileNotFoundError:
            pass  # A profile never started has no service definition.

    def restart(self, name: str) -> None:
        from hermes_cli.service_manager import _write_gateway_desired_state
        self._profile(name)
        _write_gateway_desired_state(name, 'running')
        sprites_api.request('POST', f'/services/{name}/restart')

    def is_running(self, name: str) -> bool:
        self._profile(name)
        try:
            service = sprites_api.request('GET', f'/services/{name}')
            return service.get('state', {}).get('status') == 'running'
        except FileNotFoundError:
            return False

    def supports_runtime_registration(self) -> bool:
        return True

    def register_profile_gateway(self, profile: str, *, extra_env: dict[str, str] | None = None, start_now: bool = True) -> None:
        from hermes_cli.service_manager import validate_profile_name
        validate_profile_name(profile)
        # Native PUT starts immediately. Defer registration until explicit start,
        # rather than briefly running a newly created, deliberately stopped profile.
        if not start_now:
            return
        if extra_env:
            raise ValueError('Sprites uses the protected instance environment')
        name = f'gateway-{profile}'
        try:
            sprites_api.request('GET', f'/services/{name}')
        except FileNotFoundError:
            # PUT creates (409 on an existing service); never replace a stopped service.
            sprites_api.request('PUT', f'/services/{name}', {
                'cmd': '/usr/bin/sudo',
                'args': ['-n', '/usr/bin/python3', '/opt/hermes/scripts/sprites/runtime.py', 'run', 'gateway', profile],
            })

    def unregister_profile_gateway(self, profile: str) -> None:
        from hermes_cli.service_manager import validate_profile_name
        validate_profile_name(profile)
        try:
            sprites_api.request('DELETE', f'/services/gateway-{profile}')
        except FileNotFoundError:
            pass

    def list_profile_gateways(self) -> list[str]:
        return [s['name'][8:] for s in sprites_api.request('GET', '/services') if s['name'].startswith('gateway-')]
