from idlelib.delegator import Delegator
from idlelib.redirector import WidgetRedirector


class Percolator:

    def __init__(self, text):
        # XXX would be nice to inherit from Delegator
        self.text = text
        self.redir = WidgetRedirector(text)
        self.top = self.bottom = Delegator(text)
        self.bottom.insert = self.redir.register("insert", self.insert)
        self.bottom.delete = self.redir.register("delete", self.delete)
        self.filters = []

    def close(self):
        while self.top is not self.bottom:
            self.removefilter(self.top)
        self.top = None
        self.bottom.setdelegate(None)
        self.bottom = None
        self.redir.close()
        self.redir = None
        self.text = None

    def insert(self, index, chars, tags=None):
        # Could go away if inheriting from Delegator
        self.top.insert(index, chars, tags)

    def delete(self, index1, index2=None):
        # Could go away if inheriting from Delegator
        self.top.delete(index1, index2)

    def insertfilter(self, filter):
        # Perhaps rename to pushfilter()?
        assert isinstance(filter, Delegator)
        assert filter.delegate is None
        filter.setdelegate(self.top)
        self.top = filter

    def insertfilterafter(self, filter, after):
        assert isinstance(filter, Delegator)
        assert isinstance(after, Delegator)
        assert filter.delegate is None

        f = self.top
        f.resetcache()
        while f is not after:
            assert f is not self.bottom
            f = f.delegate
            f.resetcache()

        filter.setdelegate(f.delegate)
        f.setdelegate(filter)

    def removefilter(self, filter):
        # XXX Perhaps should only support popfilter()?
        assert isinstance(filter, Delegator)
        assert filter.delegate is not None
        f = self.top
        if f is filter:
            self.top = filter.delegate
            filter.setdelegate(None)
        else:
            while f.delegate is not filter:
                assert f is not self.bottom
                f.resetcache()
                f = f.delegate
            f.setdelegate(filter.delegate)
            filter.setdelegate(None)


def _percolator(parent):  # htest #
    import tkinter as tk

    class Tracer(Delegator):
        def __init__(self, name):
            self.name = name
            Delegator.__init__(self, None)

        def insert(self, *args):
            print(self.name, ": insert", args)
            self.delegate.insert(*args)

        def delete(self, *args):
            print(self.name, ": delete", args)
            self.delegate.delete(*args)

    top = tk.Toplevel(parent)
    top.title("Test Percolator")
    x, y = map(int, parent.geometry().split('+')[1:])
    top.geometry("+%d+%d" % (x, y + 175))
    text = tk.Text(top)
    p = Percolator(text)
    pin = p.insertfilter
    pout = p.removefilter
    t1 = Tracer("t1")
    t2 = Tracer("t2")

    def toggle1():
        (pin if var1.get() else pout)(t1)
    def toggle2():
        (pin if var2.get() else pout)(t2)

    text.pack()
    text.focus_set()
    var1 = tk.IntVar(parent)
    cb1 = tk.Checkbutton(top, text="Tracer1", command=toggle1, variable=var1)
    cb1.pack()
    var2 = tk.IntVar(parent)
    cb2 = tk.Checkbutton(top, text="Tracer2", command=toggle2, variable=var2)
    cb2.pack()


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_percolator', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(_percolator)
