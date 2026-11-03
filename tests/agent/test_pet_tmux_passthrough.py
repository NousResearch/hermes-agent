"""Kitty frame chunks survive tmux passthrough without changing other encoders."""

from agent.pet import render


def test_kitty_chunks_pass_through_tmux_individually(monkeypatch):
    data = "A" * 4096 + "B" * 4096 + "C" * 8
    monkeypatch.delenv("TMUX", raising=False)
    plain = render._kitty_apc("a=T,f=100", data)
    assert plain == (
        "\x1b_Ga=T,f=100,m=1;" + "A" * 4096 + "\x1b\\"
        "\x1b_Gm=1;" + "B" * 4096 + "\x1b\\"
        "\x1b_Gm=0;" + "C" * 8 + "\x1b\\"
    )

    monkeypatch.setenv("TMUX", "/tmux/socket,1,0")
    wrapped = render._kitty_apc("a=T,f=100", data)
    chunks = ["\x1b_G" + part for part in plain.split("\x1b_G")[1:]]
    expected = "".join(f"\x1bPtmux;{chunk.replace(chr(27), chr(27) * 2)}\x1b\\" for chunk in chunks)
    assert wrapped == expected


def test_tmux_only_wraps_kitty_apc(monkeypatch):
    from PIL import Image

    frame = Image.new("RGBA", (8, 16), (20, 60, 120, 255))
    monkeypatch.delenv("TMUX", raising=False)
    virtual = render._encode_kitty_virtual(frame, image_id=7, cols=1, rows=1)
    direct = render._encode_kitty(frame)
    iterm = render._encode_iterm(frame)
    sixel = render._encode_sixel(frame)
    unicode = render._encode_unicode(frame, target_cols=4)

    monkeypatch.setenv("TMUX", "/tmux/socket,1,0")
    assert render._encode_kitty_virtual(frame, image_id=7, cols=1, rows=1) == f"\x1bPtmux;{virtual.replace(chr(27), chr(27) * 2)}\x1b\\"
    assert render._encode_kitty(frame) == f"\x1bPtmux;{direct.replace(chr(27), chr(27) * 2)}\x1b\\"
    assert render._encode_iterm(frame) == iterm
    assert render._encode_sixel(frame) == sixel
    assert render._encode_unicode(frame, target_cols=4) == unicode
    assert all("\x1b" not in row for row in render.kitty_placeholder_rows(2, 2))
