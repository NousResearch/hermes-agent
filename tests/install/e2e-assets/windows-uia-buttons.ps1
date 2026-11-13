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
