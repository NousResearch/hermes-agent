from agent.ascii_canvas import Canvas, render_table, render_tree, render_title, display_width


def test_canvas_draws_rectangle_with_merged_crossing_lines():
    canvas = Canvas(12, 7)
    canvas.rectangle(1, 1, 10, 5)
    canvas.horizontal_line(0, 3, 11)
    canvas.vertical_line(6, 0, 6)

    assert canvas.render() == "\n".join(
        [
            "      │     ",
            " ┌────┼───┐ ",
            " │    │   │ ",
            "─┼────┼───┼─",
            " │    │   │ ",
            " └────┼───┘ ",
            "      │     ",
        ]
    )


def test_canvas_draws_straight_arrows():
    canvas = Canvas(12, 5)
    canvas.arrow(1, 1, 9, 1)
    canvas.arrow(10, 3, 10, 0)

    assert canvas.render() == "\n".join(
        [
            "          ↑ ",
            " ────────>│ ",
            "          │ ",
            "          │ ",
            "            ",
        ]
    )


def test_canvas_draws_routed_arrow_with_corner_merge():
    canvas = Canvas(8, 5)
    canvas.arrow(1, 1, 5, 3)

    assert canvas.render() == "\n".join(
        [
            "        ",
            " ────┐  ",
            "     │  ",
            "     ↓  ",
            "        ",
        ]
    )


def test_render_table_golden_snapshot():
    assert render_table(
        ["Name", "Role"],
        [["EMA", "front office"], ["Gond", "engineering"]],
    ) == "\n".join(
        [
            "┌───────┬──────────────┐",
            "│ Name  │ Role         │",
            "├───────┼──────────────┤",
            "│ EMA   │ front office │",
            "│ Gond  │ engineering  │",
            "└───────┴──────────────┘",
        ]
    )


def test_render_tree_golden_snapshot():
    tree = {
        "Hermes": {
            "agent": ["tools", "skills"],
            "gateway": ["telegram"],
        }
    }

    assert render_tree(tree) == "\n".join(
        [
            "Hermes",
            "├─ agent",
            "│  ├─ tools",
            "│  └─ skills",
            "└─ gateway",
            "   └─ telegram",
        ]
    )


def test_unicode_width_policy_is_explicit_and_stable():
    assert display_width("AΩ界é") == 6

    canvas = Canvas(8, 2)
    canvas.text(0, 0, "A界B")
    canvas.text(0, 1, "abcdefghi")

    assert canvas.render() == "\n".join(["A界B    ", "abcdefgh"])


def test_figlet_title_hook_is_guarded_without_dependency_requirement():
    # The hook must never require pyfiglet to be installed. Without it, title rendering
    # falls back to plain text; with it, callers can opt into the optional renderer.
    assert isinstance(render_title("Spearhead", figlet=True), str)
    assert render_title("Spearhead", figlet=False) == "Spearhead"
