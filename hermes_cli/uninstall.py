"""
Hermes Agent 제거 도구.

제공하는 옵션:
- 전체 제거: 설정과 데이터까지 모두 삭제
- 데이터 유지: 코드는 지우고 ~/.hermes/(설정, 세션, 로그)는 유지
"""

import os
import shutil
import subprocess
from pathlib import Path

from hermes_constants import get_hermes_home

from hermes_cli.colors import Colors, color

def log_info(msg: str):
    print(f"{color('→', Colors.CYAN)} {msg}")

def log_success(msg: str):
    print(f"{color('✓', Colors.GREEN)} {msg}")

def log_warn(msg: str):
    print(f"{color('⚠', Colors.YELLOW)} {msg}")

def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()


def find_shell_configs() -> list:
    """Find shell configuration files that might have PATH entries."""
    home = Path.home()
    configs = []
    
    candidates = [
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
        home / ".zshrc",
        home / ".zprofile",
    ]
    
    for config in candidates:
        if config.exists():
            configs.append(config)
    
    return configs


def remove_path_from_shell_configs():
    """Remove Hermes PATH entries from shell configuration files."""
    configs = find_shell_configs()
    removed_from = []
    
    for config_path in configs:
        try:
            content = config_path.read_text()
            original_content = content
            
            # Remove lines containing hermes-agent or hermes PATH entries
            new_lines = []
            skip_next = False
            
            for line in content.split('\n'):
                # Skip the "# Hermes Agent" comment and following line
                if '# Hermes Agent' in line or '# hermes-agent' in line:
                    skip_next = True
                    continue
                if skip_next and ('hermes' in line.lower() and 'PATH' in line):
                    skip_next = False
                    continue
                skip_next = False
                
                # Remove any PATH line containing hermes
                if 'hermes' in line.lower() and ('PATH=' in line or 'path=' in line.lower()):
                    continue
                    
                new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            
            # Clean up multiple blank lines
            while '\n\n\n' in new_content:
                new_content = new_content.replace('\n\n\n', '\n\n')
            
            if new_content != original_content:
                config_path.write_text(new_content)
                removed_from.append(config_path)
                
        except Exception as e:
            log_warn(f"Could not update {config_path}: {e}")
    
    return removed_from


def remove_wrapper_script():
    """Remove the hermes wrapper script if it exists."""
    wrapper_paths = [
        Path.home() / ".local" / "bin" / "hermes",
        Path("/usr/local/bin/hermes"),
    ]
    
    removed = []
    for wrapper in wrapper_paths:
        if wrapper.exists():
            try:
                # Check if it's our wrapper (contains hermes_cli reference)
                content = wrapper.read_text()
                if 'hermes_cli' in content or 'hermes-agent' in content:
                    wrapper.unlink()
                    removed.append(wrapper)
            except Exception as e:
                log_warn(f"{wrapper} 를 제거하지 못했어요: {e}")
    
    return removed


def uninstall_gateway_service():
    """Stop and uninstall the gateway service if running."""
    import platform
    
    if platform.system() != "Linux":
        return False

    prefix = os.getenv("PREFIX", "")
    if os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix:
        return False
    
    try:
        from hermes_cli.gateway import get_service_name
        svc_name = get_service_name()
    except Exception:
        svc_name = "hermes-gateway"

    service_file = Path.home() / ".config" / "systemd" / "user" / f"{svc_name}.service"
    
    if not service_file.exists():
        return False
    
    try:
        # Stop the service
        subprocess.run(
            ["systemctl", "--user", "stop", svc_name],
            capture_output=True,
            check=False
        )
        
        # Disable the service
        subprocess.run(
            ["systemctl", "--user", "disable", svc_name],
            capture_output=True,
            check=False
        )
        
        # Remove service file
        service_file.unlink()
        
        # Reload systemd
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            check=False
        )
        
        return True
        
    except Exception as e:
        log_warn(f"게이트웨이 서비스를 완전히 제거하지 못했어요: {e}")
        return False


def run_uninstall(args):
    """
    Run the uninstall process.
    
    Options:
    - Full uninstall: removes code + ~/.hermes/ (configs, data, logs)
    - Keep data: removes code but keeps ~/.hermes/ for future reinstall
    """
    project_root = get_project_root()
    hermes_home = get_hermes_home()
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA, Colors.BOLD))
    print(color("│               ⚕ Hermes Agent 제거 도구                │", Colors.MAGENTA, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA, Colors.BOLD))
    print()
    
    # Show what will be affected
    print(color("현재 설치 상태:", Colors.CYAN, Colors.BOLD))
    print(f"  코드:     {project_root}")
    print(f"  설정:     {hermes_home / 'config.yaml'}")
    print(f"  시크릿:   {hermes_home / '.env'}")
    print(f"  데이터:   {hermes_home / 'cron/'}, {hermes_home / 'sessions/'}, {hermes_home / 'logs/'}")
    print()
    
    # Ask for confirmation
    print(color("제거 옵션:", Colors.YELLOW, Colors.BOLD))
    print()
    print("  1) " + color("데이터 유지", Colors.GREEN) + " - 코드만 제거하고 설정/세션/로그는 유지")
    print("     (권장 - 나중에 현재 설정을 그대로 사용해 다시 설치할 수 있어요)")
    print()
    print("  2) " + color("전체 제거", Colors.RED) + " - 데이터까지 포함해 모든 항목 제거")
    print("     (경고: 설정, 세션, 로그를 영구적으로 삭제해요)")
    print()
    print("  3) " + color("취소", Colors.CYAN) + " - 제거하지 않기")
    print()
    
    try:
        choice = input(color("옵션 선택 [1/2/3]: ", Colors.BOLD)).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print("취소했어요.")
        return
    
    if choice == "3" or choice.lower() in ("c", "cancel", "q", "quit", "n", "no"):
        print()
        print("제거를 취소했어요.")
        return
    
    full_uninstall = (choice == "2")
    
    # Final confirmation
    print()
    if full_uninstall:
        print(color("⚠️  경고: Hermes 데이터를 전부 영구적으로 삭제해요!", Colors.RED, Colors.BOLD))
        print(color("   포함 항목: 설정, API key, 세션, 예약 작업, 로그", Colors.RED))
    else:
        print("Hermes 코드는 제거하지만 설정과 데이터는 유지해요.")
    
    print()
    try:
        confirm = input(f"확인하려면 '{color('yes', Colors.YELLOW)}' 를 입력하세요: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print("취소했어요.")
        return
    
    if confirm != "yes":
        print()
        print("제거를 취소했어요.")
        return
    
    print()
    print(color("제거하는 중...", Colors.CYAN, Colors.BOLD))
    print()
    
    # 1. Stop and uninstall gateway service
    log_info("게이트웨이 서비스를 확인하는 중...")
    if uninstall_gateway_service():
        log_success("게이트웨이 서비스를 중지하고 제거했어요")
    else:
        log_info("게이트웨이 서비스를 찾지 못했어요")
    
    # 2. Remove PATH entries from shell configs
    log_info("셸 설정에서 PATH 항목을 제거하는 중...")
    removed_configs = remove_path_from_shell_configs()
    if removed_configs:
        for config in removed_configs:
            log_success(f"업데이트했어요: {config}")
    else:
        log_info("제거할 PATH 항목이 없어요")
    
    # 3. Remove wrapper script
    log_info("hermes 명령 래퍼를 제거하는 중...")
    removed_wrappers = remove_wrapper_script()
    if removed_wrappers:
        for wrapper in removed_wrappers:
            log_success(f"제거했어요: {wrapper}")
    else:
        log_info("래퍼 스크립트를 찾지 못했어요")
    
    # 4. Remove installation directory (code)
    log_info("설치 디렉터리를 제거하는 중...")
    
    # Check if we're running from within the install dir
    # We need to be careful here
    try:
        if project_root.exists():
            # If the install is inside ~/.hermes/, just remove the hermes-agent subdir
            if hermes_home in project_root.parents or project_root.parent == hermes_home:
                shutil.rmtree(project_root)
                log_success(f"제거했어요: {project_root}")
            else:
                # Installation is somewhere else entirely
                shutil.rmtree(project_root)
                log_success(f"제거했어요: {project_root}")
    except Exception as e:
        log_warn(f"{project_root} 를 완전히 제거하지 못했어요: {e}")
        log_info("수동으로 제거해야 할 수도 있어요")
    
    # 5. Optionally remove ~/.hermes/ data directory
    if full_uninstall:
        log_info("설정과 데이터를 제거하는 중...")
        try:
            if hermes_home.exists():
                shutil.rmtree(hermes_home)
                log_success(f"제거했어요: {hermes_home}")
        except Exception as e:
            log_warn(f"{hermes_home} 를 완전히 제거하지 못했어요: {e}")
            log_info("수동으로 제거해야 할 수도 있어요")
    else:
        log_info(f"설정과 데이터는 {hermes_home} 에 그대로 유지해요")
    
    # Done
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.GREEN, Colors.BOLD))
    print(color("│                   ✓ 제거 완료!                         │", Colors.GREEN, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.GREEN, Colors.BOLD))
    print()
    
    if not full_uninstall:
        print(color("설정과 데이터는 그대로 보존했어요:", Colors.CYAN))
        print(f"  {hermes_home}/")
        print()
        print("나중에 기존 설정으로 다시 설치하려면:")
        print(color("  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash", Colors.DIM))
        print()
    
    print(color("과정을 마무리하려면 셸을 다시 불러오세요:", Colors.YELLOW))
    print("  source ~/.bashrc  # 또는 ~/.zshrc")
    print()
    print("Hermes Agent를 사용해 주셔서 고마워요! ⚕")
    print()
