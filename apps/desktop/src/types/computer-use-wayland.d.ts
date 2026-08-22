import '@/types/hermes'

declare module '@/types/hermes' {
  interface ComputerUseStatus {
    /** Linux-only native Wayland/session/package diagnosis. */
    linux_wayland?: {
      distribution?: {
        id?: string | null
        id_like?: string[]
        name?: string | null
        pretty_name?: string | null
        arch_like?: boolean
      }
      session?: { kind?: string; desktop?: string | null }
      native_wayland_enabled?: boolean
      driver_features?: {
        wayland_native?: boolean
        portal_input?: boolean
        portal_capture?: boolean
        manifest_supported?: boolean
      }
      arch_packages?: { applicable?: boolean; reason?: string | null }
      selected_portal_package?: string | null
      missing_packages?: string[]
      portal_dbus_available?: boolean
      atspi_dbus_available?: boolean
      pipewire_service?: boolean
      capabilities?: {
        capture_path?: string | null
        input_path?: string | null
        activation_path?: string | null
        foreground_pointer_input?: boolean
        foreground_keyboard_input?: boolean
        consent_expected?: boolean
        restore_token_present?: boolean
        degraded_reasons?: string[]
        hard_failures?: string[]
      }
    }
    remediation?: string
  }
}
