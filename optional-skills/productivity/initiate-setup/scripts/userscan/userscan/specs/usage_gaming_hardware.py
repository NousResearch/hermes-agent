"""Hardware and health probes (Windows).

Registration only at import time. Shared helpers live in usage_gaming.
"""
from __future__ import annotations

import collections
import datetime as dt
import os
import re
import time

from userscan.registry import probe, ps_probe
from userscan.specs.usage_gaming import (_EVT_NS, _base, _evt_fields, _ex, _ft, _iso, _mtime, _paths, _read,
    _reg_keys, _reg_values, _unmatch, _wevt)

# ================================================================= HARDWARE

@probe(id="hw.system", level="L1", family="hardware", tier="T0", collect="core")
def hw_system(h, facts):
    """SMBIOS system/board/BIOS from registry, visible RAM (GlobalMemoryStatusEx), fixed-drive free space."""
    import ctypes
    b = _reg_values(r"HKLM\HARDWARE\DESCRIPTION\System\BIOS", 100)

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]
    ms = MEMSTAT()
    ms.dwLength = ctypes.sizeof(ms)
    ram = None
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        ram = round(ms.total / 2 ** 30, 1)
    drives = []
    k32 = ctypes.windll.kernel32
    mask = k32.GetLogicalDrives()
    for i in range(26):
        if not mask & (1 << i):
            continue
        root = chr(65 + i) + ":\\"
        if k32.GetDriveTypeW(ctypes.c_wchar_p(root)) != 3:
            continue
        free, total = ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)
        if k32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(root), None, ctypes.byref(total), ctypes.byref(free)) and total.value:
            drives.append([root[:2], round(total.value / 1e9, 1), round(free.value / 1e9, 1),
                           round(100 * free.value / total.value, 1)])
    return {"present": True, "manufacturer": b.get("SystemManufacturer"), "product": b.get("SystemProductName"),
            "family": b.get("SystemFamily"), "board_vendor": b.get("BaseBoardManufacturer"),
            "board": b.get("BaseBoardProduct"), "bios_vendor": b.get("BIOSVendor"), "bios_version": b.get("BIOSVersion"),
            "bios_date": b.get("BIOSReleaseDate"), "visible_ram_gb": ram, "memory_load_pct": ms.dwMemoryLoad if ram else None,
            "fixed_drives_total_free_gb_pct": drives}


@probe(id="hw.cpu", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system")
def hw_cpu(h, facts):
    """CPU name, vendor, MHz and logical processor count from HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor."""
    base = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor"
    cores = _reg_keys(h, base, 1024)
    v = _reg_values(base + "\\0", 50)
    if not v:
        return {"present": False}
    return {"present": True, "name": (v.get("ProcessorNameString") or "").strip(), "vendor": v.get("VendorIdentifier"),
            "mhz": v.get("~MHz"), "logical_processors": len(cores), "native_arch": h.l0.get("native_arch")}


@probe(id="hw.gpu", level="L1", family="hardware", tier="T0", collect="core")
def hw_gpu(h, facts):
    """Display adapters from the Display class registry key: name, driver version, VRAM (qwMemorySize)."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    out = []
    for sk in _reg_keys(h, base, 50):
        if not sk.isdigit():
            continue
        v = _reg_values(base + "\\" + sk, 400)
        name = v.get("DriverDesc")
        if not name:
            continue
        mem = v.get("HardwareInformation.qwMemorySize")
        if mem is None:
            m2 = v.get("HardwareInformation.MemorySize")
            mem = int.from_bytes(m2[:8], "little") if isinstance(m2, (bytes, bytearray)) else m2
        out.append({"name": name, "driver": v.get("DriverVersion"), "driver_date": v.get("DriverDate"),
                    "vram_gb": round(mem / 2 ** 30, 1) if isinstance(mem, int) and mem > 0 else None,
                    "vendor": v.get("ProviderName")})
    real = [g for g in out if not re.search(r"Microsoft Basic|Remote Display|Virtual|Parsec|Meta Virtual", g["name"], re.I)]
    vr = [g["vram_gb"] for g in real if g["vram_gb"]]
    return {"present": True, "max_vram_gb": max(vr) if vr else None, "adapters": real, "other": len(out) - len(real)} if out else None


@probe(id="hw.power_plan", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system")
def hw_power_plan(h, facts):
    """Active power scheme GUID and friendly name from registry; Modern Standby flag."""
    known = {"381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
             "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver", "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate Performance"}
    base = r"HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
    g = h.reg(base, "ActivePowerScheme")
    if not g:
        return {"present": False}
    fname = h.reg(base + "\\" + g, "FriendlyName")
    if fname and fname.startswith("@"):
        fname = None
    ov = {"ded574b5-45a0-4f42-8737-46345c09c238": "Best performance", "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
          "00000000-0000-0000-0000-000000000000": "Balanced"}
    ac = h.reg(base, "ActiveOverlayAcPowerScheme")
    cs = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Power", "CsEnabled")
    return {"present": True, "guid": g, "name": known.get(g.lower()) or fname or "custom",
            "overlay_ac": ov.get((ac or "").lower(), ac), "modern_standby": cs}


@probe(id="hw.battery", level="L1", family="hardware", tier="T0", collect="core")
def hw_battery(h, facts):
    """Battery present (GetSystemPowerStatus + ACPI PNP0C0A); laptop vs desktop."""
    import ctypes

    class SPS(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = SPS()
    ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    acpi = bool(_reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\ACPI\PNP0C0A", 10))
    has = acpi or (ok and s.flag not in (128, 255))
    if not has:
        return None
    return {"present": True, "on_ac": s.ac == 1 if ok else None, "charge_pct": s.pct if ok and s.pct <= 100 else None,
            "battery_saver": bool(s.saver) if ok else None}


@probe(id="boot.fastboot", level="L1", family="hardware", tier="T0", collect="core")
def boot_fastboot(h, facts):
    """Fast Startup (HiberbootEnabled): needed to read 'shutdown' events correctly."""
    v = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power", "HiberbootEnabled")
    hib = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Power", "HibernateEnabled")
    if v is None and hib is None:
        return None
    return {"present": True, "fast_startup": v, "hibernate_enabled": hib}


@probe(id="tuning.afterburner", level="L1", family="hardware", tier="T0", collect="core")
def tuning_afterburner(h, facts):
    """MSI Afterburner: autostart, fan curve, logging, per-GPU profiles (power limit, offsets, VF curve)."""
    ab = os.path.join(_paths()["PF86"], "MSI Afterburner")
    if not _ex(ab):
        return None
    out = {"present": True}
    t = _read(os.path.join(ab, "Profiles", "MSIAfterburner.cfg"), 500_000)
    if t:
        def k(n):
            m = re.search(rf"^{n}=(.*)$", t, re.M)
            return m.group(1).strip() if m else None
        out.update(start_with_windows=k("StartWithWindows"), sw_fan_curve=k("SwAutoFanControl"),
                   hw_log=k("EnableHwMonitoringLog") or k("EnableLog"))
    gpus = []
    for f in h.list_dir(os.path.join(ab, "Profiles"), 100):
        if f.startswith("VEN_") and not f.startswith("VEN_0000"):
            tt = _read(os.path.join(ab, "Profiles", f), 500_000) or ""

            def g(n):
                m = re.search(rf"^{n}=(\S+)", tt, re.M)
                return m.group(1) if m else None
            gpus.append({"vendor": {"10DE": "NVIDIA", "8086": "Intel", "1002": "AMD"}.get(f[4:8], f[4:8]), "dev": f[13:17],
                         "power_limit": g("PowerLimit"), "core_offset": g("CoreClkBoost"), "fan_mode": g("FanMode"),
                         "vf_curve": bool(g("VFCurve"))})
    out["gpu_profiles"] = gpus
    out["version"] = (_unmatch(h, r"MSI Afterburner") or [{}])[0].get("version")
    return out


@probe(id="tuning.rtss", level="L1", family="hardware", tier="T0", collect="core")
def tuning_rtss(h, facts):
    """RivaTuner Statistics Server presence and per-game profile count."""
    rt = os.path.join(_paths()["PF86"], "RivaTuner Statistics Server")
    if not _ex(rt):
        return None
    prof = [f for f in h.list_dir(os.path.join(rt, "Profiles"), 2000) if f.endswith(".cfg") and f.lower() != "global"]
    return {"present": True, "profiles": len(prof)}


@probe(id="periph.rgb_suites", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system")
def periph_rgb_suites(h, facts):
    """RGB / peripheral vendor suites from the uninstall index (Razer, Logi, Corsair, ASUS Aura, OpenRGB, ...)."""
    hits = _unmatch(h, r"Razer|Logi|G HUB|SteelSeries|Corsair|iCUE|HyperX|NGENUITY|Patriot Viper|Armoury|\bAura\b|OpenRGB|"
                       r"SignalRGB|Wallpaper Engine|DS4Windows|8BitDo|Xbox Accessories|Stream Deck|Elgato|Wooting|NZXT")
    names = sorted({re.sub(r"\s+[\d.]+$", "", x["name"]) for x in hits})
    return {"present": bool(names), "count": len(hits), "suites": names[:25]}


@probe(id="hw.audio", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.system")
def hw_audio(h, facts):
    """Audio endpoints from MMDevices registry: active render/capture counts and adapter interface names."""
    base = r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio"
    out = {"present": False}
    for kind in ("Render", "Capture"):
        keys = _reg_keys(h, base + "\\" + kind, 300)
        active = []
        for k in keys:
            st = h.reg(base + "\\" + kind + "\\" + k, "DeviceState")
            if st == 1:
                iface = h.reg(base + "\\" + kind + "\\" + k + "\\Properties", "{b3f8fa53-0004-438e-9003-51a46e139bfc},6")
                active.append(re.sub(r"(?i)\b[A-Z][a-z]+'s\b", "<name>'s", iface or "?"))
        out[kind.lower()] = {"total": len(keys), "active": len(active), "active_ifaces": sorted(set(active))[:12]}
        out["present"] = out["present"] or bool(keys)
    return out


@probe(id="hw.bluetooth", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.system")
def hw_bluetooth(h, facts):
    """Paired Bluetooth devices by class (gamepad/audio/HID) from BTHPORT registry; device names not emitted."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Services\BTHPORT\Parameters\Devices"
    keys = _reg_keys(h, base, 500)
    major = {0: "Misc", 1: "Computer", 2: "Phone", 3: "LAN", 4: "Audio/Video", 5: "Peripheral", 6: "Imaging",
             7: "Wearable", 8: "Toy", 9: "Health", 31: "Uncategorized"}
    cls = collections.Counter()
    for k in keys:
        cod = h.reg(base + "\\" + k, "COD") or 0
        if isinstance(cod, bytes):
            cod = int.from_bytes(cod[:4], "little")
        maj, mn = (cod >> 8) & 0x1F, (cod >> 2) & 0x3F
        sub = None
        if maj == 5:
            sub = {0x10: "Keyboard", 0x20: "Mouse", 0x30: "Keyboard+Mouse", 0x01: "Joystick", 0x02: "Gamepad"}.get(mn & 0x33)
        elif maj == 4:
            sub = {1: "Headset", 2: "Handsfree", 4: "Microphone", 5: "Loudspeaker", 6: "Headphones"}.get(mn)
        if not sub:
            raw = h.reg(base + "\\" + k, "Name")
            nm = raw.decode("utf-8", "replace").strip("\0") if isinstance(raw, bytes) else ""
            for rx, lbl in ((r"(?i)controller|gamepad|dualsense|dualshock|joy-?con", "Gamepad"),
                            (r"(?i)mouse|mx master|deathadder|mx anywhere", "Mouse"), (r"(?i)keyboard|keys\b", "Keyboard"),
                            (r"(?i)WH-|WF-|buds|airpods|headphone|headset", "Headphones"), (r"(?i)speaker|soundbar|kanto", "Speaker")):
                if re.search(rx, nm):
                    sub = lbl + "(name)"
                    break
        cls[major.get(maj, str(maj)) + ("/" + sub if sub else "")] += 1
    le = sum(len(_reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\BTHLE\\" + d, 50))
             for d in _reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\BTHLE", 300))
    if not keys and not le:
        return {"present": False}
    return {"present": True, "classic_paired": len(keys), "ble_instances": le, "by_class": dict(cls)}


@probe(id="hw.usb_history", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.system")
def hw_usb_history(h, facts):
    """USB device ids/instances ever seen and USB storage count (Enum\\USB, USBSTOR); top vendor ids only."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Enum\USB"
    ids = _reg_keys(h, base, 2000)
    inst = 0
    vids = collections.Counter()
    for i in ids:
        inst += len(_reg_keys(h, base + "\\" + i, 200))
        m = re.search(r"VID_([0-9A-Fa-f]{4})", i)
        if m:
            vids[m.group(1).upper()] += 1
    stor = sum(len(_reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\USBSTOR\\" + d, 100))
               for d in _reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\USBSTOR", 500))
    vend = {"046D": "Logitech", "1532": "Razer", "1B1C": "Corsair", "1038": "SteelSeries", "0B05": "ASUS", "045E": "Microsoft",
            "054C": "Sony", "057E": "Nintendo", "28DE": "Valve", "3434": "Keychron", "0951": "Kingston/HyperX", "03F0": "HP/HyperX",
            "8087": "Intel", "0BDA": "Realtek", "0955": "NVIDIA", "1462": "MSI", "05AC": "Apple", "04E8": "Samsung", "0781": "SanDisk"}
    if not ids:
        return {"present": False}
    return {"present": True, "device_ids_ever": len(ids), "instances_ever": inst, "usbstor_instances": stor,
            "top_vendors": [[v, vend.get(v), n] for v, n in vids.most_common(12)]}


@probe(id="hw.monitor_mode", level="L2", family="hardware", tier="T0", collect="extended", gate="hw.system")
def hw_monitor_mode(h, facts):
    """Current resolution/refresh per display (EnumDisplaySettings); falls back to GraphicsDrivers\\Configuration."""
    import ctypes
    from ctypes import wintypes as W

    class DEVMODEW(ctypes.Structure):
        _fields_ = [("dmDeviceName", ctypes.c_wchar * 32), ("dmSpecVersion", W.WORD), ("dmDriverVersion", W.WORD),
                    ("dmSize", W.WORD), ("dmDriverExtra", W.WORD), ("dmFields", W.DWORD), ("dmPositionX", W.LONG),
                    ("dmPositionY", W.LONG), ("dmDisplayOrientation", W.DWORD), ("dmDisplayFixedOutput", W.DWORD),
                    ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                    ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short), ("dmFormName", ctypes.c_wchar * 32),
                    ("dmLogPixels", W.WORD), ("dmBitsPerPel", W.DWORD), ("dmPelsWidth", W.DWORD), ("dmPelsHeight", W.DWORD),
                    ("dmDisplayFlags", W.DWORD), ("dmDisplayFrequency", W.DWORD), ("dmICMMethod", W.DWORD),
                    ("dmICMIntent", W.DWORD), ("dmMediaType", W.DWORD), ("dmDitherType", W.DWORD), ("dmReserved1", W.DWORD),
                    ("dmReserved2", W.DWORD), ("dmPanningWidth", W.DWORD), ("dmPanningHeight", W.DWORD)]

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [("cb", W.DWORD), ("DeviceName", ctypes.c_wchar * 32), ("DeviceString", ctypes.c_wchar * 128),
                    ("StateFlags", W.DWORD), ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128)]
    u32 = ctypes.windll.user32
    modes = []
    i = 0
    while i < 16:
        dd = DISPLAY_DEVICEW()
        dd.cb = ctypes.sizeof(dd)
        if not u32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            break
        i += 1
        if not dd.StateFlags & 1:
            continue
        dm = DEVMODEW()
        dm.dmSize = ctypes.sizeof(dm)
        if u32.EnumDisplaySettingsW(dd.DeviceName, -1, ctypes.byref(dm)):
            modes.append({"w": dm.dmPelsWidth, "h": dm.dmPelsHeight, "hz": dm.dmDisplayFrequency, "adapter": dd.DeviceString})
    if modes:
        return {"present": True, "via": "EnumDisplaySettings", "displays": modes}
    base = r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\Configuration"
    best, best_ts = None, -1
    for cfg in _reg_keys(h, base, 200):
        ts = h.reg(base + "\\" + cfg, "Timestamp") or 0
        if isinstance(ts, int) and ts > best_ts:
            best, best_ts = cfg, ts
    if not best:
        return {"present": False}
    out = []
    for t in _reg_keys(h, base + "\\" + best, 20):
        for m in _reg_keys(h, base + "\\" + best + "\\" + t, 20):
            v = _reg_values(base + "\\" + best + "\\" + t + "\\" + m, 100)
            if v.get("PrimSurfSize.cx"):
                num, den = v.get("VSyncFreq.Numerator"), v.get("VSyncFreq.Denominator")
                out.append({"w": v["PrimSurfSize.cx"], "h": v.get("PrimSurfSize.cy"),
                            "hz": round(num / den, 1) if num and den else None})
    return {"present": bool(out), "via": "GraphicsDrivers\\Configuration", "committed": _iso(_ft(best_ts)), "displays": out}


@probe(id="hw.nvidia_smi", level="L2", family="hardware", tier="T0", collect="extended", gate="hw.gpu")
def hw_nvidia_smi(h, facts):
    """nvidia-smi query: VRAM, power draw/limit, temperature, pstate, driver."""
    P = _paths()
    exe = next((p for p in (os.path.join(P["WIN"], "System32", "nvidia-smi.exe"),
                            os.path.join(P["PF"], "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")) if _ex(p)), None)
    if not exe:
        return {"present": False}
    fields = "name,memory.total,power.draw,power.limit,power.default_limit,power.max_limit,temperature.gpu,pstate,driver_version,fan.speed"
    out = h.run([exe, "--query-gpu=" + fields, "--format=csv,noheader,nounits"], timeout_ms=5000, text=False)
    if not out:
        return {"present": False, "exe": True}
    gpus = []
    for line in out.strip().splitlines()[:8]:
        vals = [x.strip() for x in line.split(",")]
        gpus.append({k: (None if v in ("[N/A]", "N/A", "[Not Supported]") else v) for k, v in zip(fields.split(","), vals)})
    vram = None
    try:
        vram = round(max(float(g["memory.total"]) for g in gpus if g.get("memory.total")) / 1024, 1)
    except ValueError:
        pass
    return {"present": bool(gpus), "vram_gb": vram, "gpus": gpus}


ps_probe("hw.monitors", r"""
try {
  function U($a) { if ($a) { (($a | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ }) -join '').Trim() } }
  $ids = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID -ErrorAction SilentlyContinue | ForEach-Object {
    [ordered]@{ mfr = (U $_.ManufacturerName); product = (U $_.ProductCodeID); name = (U $_.UserFriendlyName); year = $_.YearOfManufacture; active = $_.Active } })
  $size = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.MaxHorizontalImageSize) { [math]::Round([math]::Sqrt([math]::Pow($_.MaxHorizontalImageSize,2)+[math]::Pow($_.MaxVerticalImageSize,2))/2.54,1) } })
  $tech = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorConnectionParams -ErrorAction SilentlyContinue | ForEach-Object {
    switch ([int64]$_.VideoOutputTechnology) { 0 {'VGA'} 4 {'DVI'} 5 {'HDMI'} 10 {'DP'} 11 {'DP-embedded'} 2147483648 {'Internal'} default { [string]$_.VideoOutputTechnology } } })
  [ordered]@{ present = ($ids.Count -gt 0); count = $ids.Count; monitors = $ids; diag_in = $size; outputs = $tech }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="extended", gate="hw.system", timeout_ms=6000,
         doc="Connected monitors from EDID (WmiMonitorID): maker, model, year, size, connection; no serials.")

ps_probe("hw.ram", r"""
try {
  $m = @(Get-CimInstance Win32_PhysicalMemory -ErrorAction SilentlyContinue)
  $a = @(Get-CimInstance Win32_PhysicalMemoryArray -ErrorAction SilentlyContinue)
  [ordered]@{ present = ($m.Count -gt 0); modules = $m.Count
    total_gb = [math]::Round((($m | Measure-Object Capacity -Sum).Sum) / 1GB, 1)
    slots = ($a | Measure-Object MemoryDevices -Sum).Sum
    speed = @($m | ForEach-Object { $_.Speed } | Select-Object -Unique)
    configured = @($m | ForEach-Object { $_.ConfiguredClockSpeed } | Select-Object -Unique)
    rated_mhz = ($m | Measure-Object Speed -Minimum).Minimum; configured_mhz = ($m | Measure-Object ConfiguredClockSpeed -Minimum).Minimum
    smbios_type = @($m | ForEach-Object { $_.SMBIOSMemoryType } | Select-Object -Unique)
    mfr = @($m | ForEach-Object { ([string]$_.Manufacturer).Trim() } | Select-Object -Unique)
    part = @($m | ForEach-Object { ([string]$_.PartNumber).Trim() } | Select-Object -Unique) }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="extended", gate="hw.system", timeout_ms=6000,
         doc="Installed RAM modules, slots, rated vs configured speed (Win32_PhysicalMemory).")

ps_probe("hw.disks", r"""
try {
  $pd = @(Get-PhysicalDisk -ErrorAction SilentlyContinue | ForEach-Object {
    $rc = $_ | Get-StorageReliabilityCounter -ErrorAction SilentlyContinue
    [ordered]@{ name = $_.FriendlyName; media = [string]$_.MediaType; bus = [string]$_.BusType; size_gb = [math]::Round($_.Size/1GB)
      health = [string]$_.HealthStatus; wear_pct = $rc.Wear; temp_c = $rc.Temperature; temp_max_c = $rc.TemperatureMax
      power_on_h = $rc.PowerOnHours; read_err_uncorrected = $rc.ReadErrorsUncorrected; write_err_uncorrected = $rc.WriteErrorsUncorrected } })
  $vol = @(Get-Volume -ErrorAction SilentlyContinue | Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' -and $_.Size } | ForEach-Object {
    [ordered]@{ letter = [string]$_.DriveLetter; size_gb = [math]::Round($_.Size/1GB,1); free_gb = [math]::Round($_.SizeRemaining/1GB,1)
      free_pct = [math]::Round(100*$_.SizeRemaining/$_.Size,1) } })
  [ordered]@{ present = ($pd.Count -gt 0); count = $pd.Count; disks = $pd; volumes = $vol }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="deep", gate="hw.system", timeout_ms=15000,
         doc="Physical disks with health, wear, temperature and error counters (Get-PhysicalDisk).")

ps_probe("hw.gpu_history", r"""
try {
  $g = @(Get-PnpDevice -Class Display -ErrorAction SilentlyContinue | ForEach-Object {
    $fi = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_FirstInstallDate -ErrorAction SilentlyContinue).Data
    $la = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_LastArrivalDate -ErrorAction SilentlyContinue).Data
    [ordered]@{ name = $_.FriendlyName; pci = (($_.InstanceId -split '\\')[1] -replace '&REV.*$',''); present = $_.Present
      first = $(if ($fi) { ([datetime]$fi).ToString('s') }); last = $(if ($la) { ([datetime]$la).ToString('s') }) } })
  [ordered]@{ present = ($g.Count -gt 0); ever = $g.Count; now = @($g | Where-Object { $_.present }).Count
    swaps = @($g | Where-Object { -not $_.present -and $_.name -notmatch 'Basic|Remote|Virtual' }).Count; adapters = $g }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="deep", gate="hw.gpu", timeout_ms=15000,
         doc="Every display adapter ever installed with first-install/last-arrival dates (GPU swaps).")

ps_probe("hw.monitor_history", r"""
try {
  $h = @(Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.InstanceId -match '^DISPLAY\\' -and $_.InstanceId -notmatch 'DEFAULT_MONITOR' } | ForEach-Object {
    $fi = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_FirstInstallDate -ErrorAction SilentlyContinue).Data
    $la = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_LastArrivalDate -ErrorAction SilentlyContinue).Data
    [pscustomobject]@{ pnp = ($_.InstanceId -split '\\')[1]; present = [bool]$_.Present; first = $fi; last = $la } } |
    Group-Object pnp | ForEach-Object { $f = @($_.Group | Where-Object first | Sort-Object first); $l = @($_.Group | Where-Object last | Sort-Object last -Descending)
      [ordered]@{ pnp = $_.Name; present = [bool]($_.Group | Where-Object present)
        first = $(if ($f) { ([datetime]$f[0].first).ToString('s') }); last = $(if ($l) { ([datetime]$l[0].last).ToString('s') }) } })
  [ordered]@{ present = ($h.Count -gt 0); distinct_monitors_ever = $h.Count; changes = @($h | Where-Object { -not $_.present }).Count; monitors = $h }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="deep", gate="hw.system", timeout_ms=15000,
         doc="Every monitor model (PnP id) ever connected with first/last dates.")

ps_probe("hw.peripherals", r"""
try {
  $cls = 'Keyboard','Mouse','HIDClass','Camera','Image','Biometric','XnaComposite','Printer','MEDIA','SmartCardReader'
  $pp = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $cls -contains $_.Class })
  $by = @{}; foreach ($d in $pp) { $by[[string]$d.Class] = 1 + [int]$by[[string]$d.Class] }
  $vids = @{}; foreach ($d in $pp) { if ($d.InstanceId -match 'VID_([0-9A-F]{4})') { $vids[$matches[1]] = 1 + [int]$vids[$matches[1]] } }
  $pads = @($pp | Where-Object { $_.Class -eq 'XnaComposite' -or $_.FriendlyName -match 'game ?controller|gamepad|xbox|dualsense|dualshock' }).Count
  $cams = @($pp | Where-Object { $_.Class -in 'Camera','Image' }).Count
  [ordered]@{ present = ($pp.Count -gt 0); by_class = $by; hid_vids = $vids; game_controllers = $pads; cameras = $cams }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T1", collect="deep", gate="hw.system", timeout_ms=15000,
         doc="Present peripherals by PnP class, USB vendor ids, controller and camera counts.")


def _xml_local(path):
    import xml.etree.ElementTree as ET
    try:
        return ET.parse(path).getroot()
    except Exception:
        return None


def _ln(el):
    return el.tag.rsplit("}", 1)[-1]


@probe(id="hw.battery_report", level="L2", family="hardware", tier="T1", collect="deep", gate="hw.battery", timeout_ms=15000)
def hw_battery_report(h, facts):
    """powercfg /batteryreport (XML into scratch): design vs full capacity, cycle count, recent AC/DC usage."""
    out = os.path.join(h.scratch(), "battery.xml")
    h.run(["powercfg", "/batteryreport", "/xml", "/output", out], timeout_ms=15000, text=False)
    root = _xml_local(out)
    if root is None:
        return {"present": False}
    bat = next((e for e in root.iter() if _ln(e) == "Battery"), None)
    if bat is None:
        return {"present": False}
    kv = {_ln(c): (c.text or "").strip() for c in bat}
    design, full = int(kv.get("DesignCapacity") or 0), int(kv.get("FullChargeCapacity") or 0)
    usage = [e for e in root.iter() if _ln(e) == "UsageEntry"]
    ac = sum(1 for e in usage if e.get("Ac") == "1")
    try:
        os.remove(out)
    except OSError:
        pass
    return {"present": True, "design_mwh": design, "full_mwh": full, "health_pct": round(100 * full / design, 1) if design else None,
            "cycles": kv.get("CycleCount"), "chemistry": kv.get("Chemistry"), "recent_usage_entries": len(usage),
            "recent_ac": ac, "recent_dc": len(usage) - ac}


@probe(id="hw.sleep_study", level="L2", family="hardware", tier="T1", collect="deep", gate="hw.battery", timeout_ms=20000)
def hw_sleep_study(h, facts):
    """powercfg /sleepstudy (XML into scratch): Modern Standby sessions, low-power share, drain."""
    out = os.path.join(h.scratch(), "sleepstudy.xml")
    h.run(["powercfg", "/sleepstudy", "/xml", "/output", out], timeout_ms=20000, text=False)
    root = _xml_local(out)
    if root is None:
        return {"present": False}
    inst = [e for e in root.iter() if _ln(e) == "ScenarioInstance"]
    types = collections.Counter(e.get("Type") or "?" for e in inst)
    lp = []
    for e in inst:
        d, l = int(e.get("Duration") or 0), int(e.get("LowPowerStateTime") or 0)
        if d:
            lp.append(100 * l / d)
    try:
        os.remove(out)
    except OSError:
        pass
    return {"present": bool(inst), "sessions": len(inst), "types": dict(types),
            "lowpower_pct_median": round(sorted(lp)[len(lp) // 2], 1) if lp else None,
            "lowpower_pct_min": round(min(lp), 1) if lp else None}


# ================================================================= HEALTH

@probe(id="health.dumps", level="L1", family="health", tier="T1", collect="core")
def health_dumps(h, facts):
    """Crash dump config (CrashControl) and dump files: Minidump count/newest, MEMORY.DMP, LiveKernelReports."""
    P = _paths()
    cc = _reg_values(r"HKLM\SYSTEM\CurrentControlSet\Control\CrashControl", 100)
    mini, newest, mini30 = 0, None, 0
    try:
        with os.scandir(os.path.join(P["WIN"], "Minidump")) as it:
            for e in it:
                if mini >= 1000:
                    break
                if e.is_file():
                    mini += 1
                    m = e.stat().st_mtime
                    newest = m if newest is None or m > newest else newest
                    mini30 += time.time() - m <= 30 * 86400
    except OSError:
        pass
    lkr = {}
    lroot = os.path.join(P["WIN"], "LiveKernelReports")
    for d in h.list_dir(lroot, 50):
        p = os.path.join(lroot, d)
        if os.path.isdir(p):
            n = sum(1 for f in h.list_dir(p, 500) if f.lower().endswith(".dmp"))
            if n:
                lkr[d] = n
        elif d.lower().endswith(".dmp"):
            lkr["root"] = lkr.get("root", 0) + 1
    md = h.meta(os.path.join(P["WIN"], "MEMORY.DMP"))
    if not cc and not mini and not lkr:
        return None
    return {"present": True, "crash_dump_enabled": cc.get("CrashDumpEnabled"), "auto_reboot": cc.get("AutoReboot"),
            "minidumps": mini, "count_30d": mini30, "minidump_newest": _iso(newest), "memory_dmp_bytes": md.get("bytes"),
            "live_kernel_reports": lkr}


_WER_NOISE = re.compile(r"(?i)^(svchost\.exe|TrustedInstaller\.exe|TiWorker\.exe|MoUsoCoreWorker\.exe|WinStore\.App\.exe|"
                        r"Microsoft\.WindowsStore.*|MicrosoftEdgeUpdate.*|MicrosoftEdge_X64_.*|setup\.exe|MsiExec\.exe|"
                        r"wermgr\.exe|StoreDesktopExtension.*|backgroundTaskHost\.exe|RuntimeBroker\.exe|SearchHost\.exe|"
                        r"MpSigStub\.exe|MsMpEng\.exe|WindowsPackageManagerServer\.exe)$")


@probe(id="health.wer", level="L2", family="health", tier="T1", collect="extended", gate="health.dumps")
def health_wer(h, facts):
    """WER report dirs: crashes/hangs per app (Report.wer AppName), kernel reports; svchost/Store noise split out."""
    roots = [os.path.join(_paths()["PD"], "Microsoft", "Windows", "WER", r) for r in ("ReportArchive", "ReportQueue")]
    types = collections.Counter()
    apps = collections.Counter()
    last = {}
    kernel = collections.Counter()
    noise = total = 0
    oldest = newest = None
    for rt in roots:
        for n in h.list_dir(rt, 1000):
            p = os.path.join(rt, n)
            total += 1
            typ = n.split("_", 1)[0]
            types[typ] += 1
            m = _mtime(p)
            if m:
                oldest = m if oldest is None or m < oldest else oldest
                newest = m if newest is None or m > newest else newest
            app, evt = None, None
            try:
                with open(os.path.join(p, "Report.wer"), "rb") as f:
                    txt = f.read(200_000).decode("utf-16", "replace")
                ma = re.search(r"^AppPath=([^\r\n]+)", txt, re.M) or re.search(r"^AppName=([^\r\n]+)", txt, re.M)
                me = re.search(r"^EventType=([^\r\n]+)", txt, re.M)
                app, evt = (_base(ma.group(1).strip()) if ma else None), (me.group(1).strip() if me else None)
            except OSError:
                pass
            if re.search(r"Kernel|BlueScreen", typ, re.I) or (evt and re.search(r"LiveKernel|BlueScreen", evt, re.I)):
                kernel[evt or typ] += 1
                continue
            if re.search(r"AppCrash|AppHang|BEX|Critical", typ, re.I):
                m2 = re.match(r"^[^_]+_([^_]+?\.exe)_", n, re.I)
                if m2:
                    app = m2.group(1)
                app = app or "?"
                if _WER_NOISE.match(app):
                    noise += 1
                    continue
                apps[app] += 1
                last[app] = max(last.get(app) or 0, m or 0)
    if not total:
        return {"present": False}
    types = collections.Counter({k: v for k, v in types.items() if not re.fullmatch(r"[0-9a-fA-F-]{36}", k)})
    return {"present": True, "reports": total, "by_type": dict(types.most_common(8)), "system_noise_crashes": noise,
            "max_repeat": apps.most_common(1)[0][1] if apps else 0,
            "crashes_by_app": [[a, c, _iso(last.get(a))] for a, c in apps.most_common(12)], "kernel_reports": dict(kernel),
            "oldest": _iso(oldest), "newest": _iso(newest)}


@probe(id="health.whea_gpu", level="L2", family="health", tier="T1", collect="extended", gate="hw.system", timeout_ms=20000)
def health_whea_gpu(h, facts):
    """Hardware error events: WHEA, nvlddmkm, display TDR 4101, bugcheck 1001 codes, storage warnings (wevtutil)."""
    provs = ("Microsoft-Windows-WHEA-Logger", "nvlddmkm", "Display", "Microsoft-Windows-WER-SystemErrorReporting",
             "disk", "Ntfs", "stornvme")
    xp = "*[System[Provider[" + " or ".join(f"@Name='{p}'" for p in provs) + "]]]"
    evs = _wevt(h, "System", xp, 3000, newest_first=True)
    if evs is None:
        return {"present": False, "error": "wevtutil failed"}
    grp = {k: {"count": 0, "by_id": collections.Counter(), "newest": None} for k in ("whea", "nvlddmkm", "tdr_4101", "storage_warn_err")}
    bug = []
    now = dt.datetime.now().astimezone()
    c30 = 0
    for ev in evs:
        try:
            prov, eid, t, data = _evt_fields(ev)
            lvl = int(ev.find(_EVT_NS + "System").find(_EVT_NS + "Level").text or 4)
        except (AttributeError, ValueError, TypeError):
            continue
        key = None
        if prov == "Microsoft-Windows-WHEA-Logger":
            key = "whea"
        elif prov == "nvlddmkm":
            key = "nvlddmkm"
        elif prov == "Display" and eid == 4101:
            key = "tdr_4101"
        elif prov in ("disk", "Ntfs", "stornvme") and 1 <= lvl <= 3:
            key = "storage_warn_err"
        elif prov == "Microsoft-Windows-WER-SystemErrorReporting" and eid == 1001:
            m = re.search(r"0x[0-9a-fA-F]+", " ".join(v or "" for v in data.values()))
            bug.append({"t": _iso(t), "code": m.group(0) if m else None})
            c30 += bool(t and (now - t).days < 30)
            continue
        if not key:
            continue
        g = grp[key]
        g["count"] += 1
        g["by_id"][str(eid)] += 1
        if t and (g["newest"] is None or _iso(t) > g["newest"]):
            g["newest"] = _iso(t)
        if key != "storage_warn_err" and t and (now - t).days < 30:
            c30 += 1
    for g in grp.values():
        g["by_id"] = dict(g["by_id"])
    return {"present": True, "count_30d": c30, **grp, "bugchecks": bug[:20]}


ps_probe("health.reliability", r"""
try {
  $r = @(Get-CimInstance Win32_ReliabilityRecords -ErrorAction SilentlyContinue)
  $m = @(Get-CimInstance Win32_ReliabilityStabilityMetrics -ErrorAction SilentlyContinue | Sort-Object TimeGenerated)
  $s = @($r | Sort-Object TimeGenerated)
  function T($grp) { $o = [ordered]@{}; foreach ($x in @($grp | Sort-Object Count -Descending | Select-Object -First 8)) { $o[[string]$x.Name] = $x.Count }; $o }
  [ordered]@{ present = ($r.Count -gt 0); records = $r.Count
    oldest = $(if ($s) { $s[0].TimeGenerated.ToString('s') }); newest = $(if ($s) { $s[-1].TimeGenerated.ToString('s') })
    by_source = (T ($r | Group-Object SourceName)); top_products = (T ($r | Where-Object ProductName | Group-Object ProductName))
    stability_latest = $(if ($m) { [math]::Round($m[-1].SystemStabilityIndex, 2) })
    stability_min = $(if ($m) { [math]::Round(($m | Measure-Object SystemStabilityIndex -Minimum).Minimum, 2) }) }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="health", tier="T1", collect="deep", gate="hw.system", timeout_ms=20000,
         doc="Reliability Monitor records by source/product and the stability index (latest, min).")
