# UI Automation button lookup for the Windows packaged-update driver.

function Find-InvokableButton {
    # Patterns are in priority order and the first pattern any button matches wins, whatever
    # the tree order: on the About page the sidebar's 'About' comes before 'Check now'.
    param($Root, [string[]]$Patterns)
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Button)
    $named = @(foreach ($b in $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)) {
        $name = ""
        try { $name = $b.Current.Name } catch {}
        if ($name) { @{ Element = $b; Name = $name } }
    })
    foreach ($pat in $Patterns) {
        foreach ($button in $named) {
            if ($button.Name -match $pat) { return $button }
        }
    }
    return $null
}

function Get-ProcessWindows {
    # Top-level windows a process draws in. App Installer 1.26 (the x64 runner's) is a UWP app:
    # its window is an ApplicationFrameHost frame, and only the frame's CoreWindow child belongs
    # to App Installer. 1.29 (WinUI desktop) owns its top-level windows itself.
    param($Root, [int]$ProcessId)
    $owned = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $ProcessId)
    $frames = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ClassNameProperty, 'ApplicationFrameWindow')
    @($Root.FindAll([System.Windows.Automation.TreeScope]::Children, $owned))
    foreach ($frame in $Root.FindAll([System.Windows.Automation.TreeScope]::Children, $frames)) {
        @($frame.FindAll([System.Windows.Automation.TreeScope]::Children, $owned))
    }
}

function Find-WindowButton {
    # Every window of every listed process: App Installer 1.29 can open its prompt in a second
    # process, beside the one the update check Windows runs when OLD launches started.
    param($Root, [int[]]$ProcessIds, [string[]]$Patterns)
    foreach ($id in $ProcessIds) {
        foreach ($window in (Get-ProcessWindows -Root $Root -ProcessId $id)) {
            $hit = Find-InvokableButton -Root $window -Patterns $Patterns
            if ($hit) { return $hit }
        }
    }
    return $null
}

function Confirm-AppInstallerPrompt {
    # Presses App Installer's prompt button, then Yes on the "Warning: Metered Connection" dialog
    # App Installer 1.26 opens over the prompt on a metered network, as the x64 runner's is.
    # Yes counts only after the prompt: a user answers the warning the prompt raised, nothing else.
    # Returns the names pressed, in order; none when no prompt appeared.
    param($Root, [scriptblock]$GetProcessIds, [scriptblock]$Press,
        [int]$PromptTimeoutSec, [int]$WarningTimeoutSec, [int]$PollSec = 3)
    $stages = @(
        @{ Patterns = @('^Update$', '^Install$', '^(Reinstall|Update) .*'); TimeoutSec = $PromptTimeoutSec },
        @{ Patterns = @('^Yes$'); TimeoutSec = $WarningTimeoutSec })
    $pressed = @()
    foreach ($stage in $stages) {
        $deadline = (Get-Date).AddSeconds($stage.TimeoutSec)
        $hit = $null
        while (-not $hit -and (Get-Date) -lt $deadline) {
            $ids = @(& $GetProcessIds)
            if ($ids.Count) { $hit = Find-WindowButton -Root $Root -ProcessIds $ids -Patterns $stage.Patterns }
            if ($hit -and -not (& $Press $hit)) { $hit = $null }
            if (-not $hit) { Start-Sleep -Seconds $PollSec }
        }
        if (-not $hit) { break }
        $pressed += $hit.Name
    }
    return $pressed
}
