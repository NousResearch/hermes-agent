from idlelib import rpc

def remote_object_tree_item(item):
    wrapper = WrappedObjectTreeItem(item)
    oid = id(wrapper)
    rpc.objecttable[oid] = wrapper
    return oid

class WrappedObjectTreeItem:
    # Lives in PYTHON subprocess

    def __init__(self, item):
        self.__item = item

    def __getattr__(self, name):
        value = getattr(self.__item, name)
        return value

    def _GetSubList(self):
        sub_list = self.__item._GetSubList()
        return list(map(remote_object_tree_item, sub_list))

class StubObjectTreeItem:
    # Lives in IDLE process

    def __init__(self, sockio, oid):
        self.sockio = sockio
        self.oid = oid

    def __getattr__(self, name):
        value = rpc.MethodProxy(self.sockio, self.oid, name)
        return value

    def _GetSubList(self):
        sub_list = self.sockio.remotecall(self.oid, "_GetSubList", (), {})
        return [StubObjectTreeItem(self.sockio, oid) for oid in sub_list]


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_debugobj_r', verbosity=2)
