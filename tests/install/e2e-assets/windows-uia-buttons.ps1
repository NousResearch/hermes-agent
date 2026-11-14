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
